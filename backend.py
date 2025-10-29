from flask import Flask, request, jsonify, session, send_from_directory, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime, timedelta
import secrets

from PIL import Image
import io
import logging

import google.generativeai as genai
import re
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

# Optional: Google Cloud Speech-to-Text
try:
    from google.cloud import speech_v1p1beta1 as speech
except Exception:
    speech = None

load_dotenv()
print(f"DEBUG: GEMINI_API_KEY loaded = {os.getenv('GEMINI_API_KEY')[:20] if os.getenv('GEMINI_API_KEY') else 'NOT FOUND'}...")

app = Flask(__name__)


_secret_key = os.getenv('SECRET_KEY')
if not _secret_key:
    logging.warning("SECRET_KEY not set; generating ephemeral key for this process")
    _secret_key = secrets.token_urlsafe(32)
app.config['SECRET_KEY'] = _secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///byte_app.db'  # Local SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['COUPONS_FILE'] = os.getenv('COUPONS_FILE', 'coupons.json')

# Session configuration for better OAuth handling
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# Google OAuth Configuration (no hardcoded defaults)
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
# Redirect URI: computed dynamically to match current host (reduce mismatch)
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', '')

def get_google_redirect_uri():
    # If provided via env, use that; otherwise use Flask url_for on current request
    if GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    try:
        from flask import url_for
        # For ngrok, always use https scheme
        if 'ngrok' in request.host:
            return url_for('google_callback', _external=True, _scheme='https')
        else:
            return url_for('google_callback', _external=True, _scheme='http')
    except Exception:
        base = request.host_url.rstrip('/')
        # Ensure https for ngrok URLs
        if 'ngrok' in base and not base.startswith('https'):
            base = base.replace('http://', 'https://')
        return f"{base}/api/auth/google/callback"

# Google Gemini Configuration (no hardcoded default)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("SUCCESS: Gemini API configured successfully")
        logging.info("Gemini API configured successfully")
    else:
        print("WARNING: GEMINI_API_KEY not set; AI responses will use fallbacks")
        logging.warning("GEMINI_API_KEY not set; AI responses will use fallbacks")
        model = None
except Exception as e:
    print(f"ERROR: Failed to configure Gemini API: {e}")
    logging.error(f"Failed to configure Gemini API: {e}")
    model = None

# OCR is handled by Gemini Vision API
print("SUCCESS: OCR available through Gemini Vision API")
logging.info("OCR available through Gemini Vision API")
ocr_reader = "gemini"  # Use Gemini for OCR

# Speech-to-Text config
SPEECH_LANGUAGE = os.getenv('SPEECH_LANGUAGE', 'en-US')

# Initialize database
db = SQLAlchemy(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure coupons file exists
def _ensure_coupons_file_exists():
    try:
        if not os.path.exists(app.config['COUPONS_FILE']):
            with open(app.config['COUPONS_FILE'], 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to ensure coupons file exists: {e}")

_ensure_coupons_file_exists()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('byte_app.log'),
        logging.StreamHandler()
    ]
)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    login_method = db.Column(db.String(20), default='form')
    school_id = db.Column(db.String(50), nullable=True)  # School ID for coupon login
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    chats = db.relationship('Chat', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'login_method': self.login_method,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat()
        }

class CouponCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    school_name = db.Column(db.String(100), nullable=False)
    school_domain = db.Column(db.String(100), nullable=False)  # Email domain restriction
    is_active = db.Column(db.Boolean, default=True)
    max_uses = db.Column(db.Integer, default=100)  # Maximum number of uses
    current_uses = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)  # Optional expiration
    
    def is_valid(self):
        """Check if coupon is still valid"""
        if not self.is_active:
            return False
        if self.current_uses >= self.max_uses:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def can_use_for_email(self, email):
        """Check if coupon can be used for this email domain"""
        email_domain = email.split('@')[1].lower() if '@' in email else ''
        return email_domain == self.school_domain.lower()

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title or f'Chat {self.id}',
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self.messages)
        }

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    sender = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default='text')
    file_path = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'sender': self.sender,
            'content': self.content,
            'message_type': self.message_type,
            'file_path': self.file_path,
            'timestamp': self.timestamp.isoformat()
        }

# NEW: To-Do Task model
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.String(500), nullable=False)
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'content': self.content,
            'is_completed': self.is_completed,
            'created_at': self.created_at.isoformat()
        }

# Per-user unique coupon generated for school ID logins
# (moved earlier in file; remove duplicate definition)

# Missing model: UserCoupon (referenced throughout auth and user endpoints)
class UserCoupon(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    code = db.Column(db.String(32), unique=True, nullable=False)
    school_name = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'code': self.code,
            'school_name': self.school_name,
            'created_at': self.created_at.isoformat()
        }

# Coupon file helpers
def validate_coupon_data(data: dict) -> tuple[bool, str]:
    try:
        code = (data.get('code') or '').strip().upper()
        school_name = (data.get('school_name') or '').strip()
        school_domain = (data.get('school_domain') or '').strip().lower()
        max_uses = int(data.get('max_uses', 100))
        is_active = bool(data.get('is_active', True))

        if not code or len(code) > 32 or not code.isalnum():
            return False, 'Invalid code format (alphanumeric, <=32)'
        if not school_name:
            return False, 'school_name is required'
        if not school_domain or '.' not in school_domain:
            return False, 'Invalid school_domain'
        if max_uses <= 0:
            return False, 'max_uses must be > 0'
        return True, ''
    except Exception:
        return False, 'Invalid coupon payload'

def load_coupons_from_file() -> list:
    try:
        with open(app.config['COUPONS_FILE'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logging.warning('coupons.json is not a list; ignoring')
                return []
            valid = []
            for item in data:
                ok, _ = validate_coupon_data(item)
                if ok:
                    # normalize
                    valid.append({
                        'code': item['code'].strip().upper(),
                        'school_name': item['school_name'].strip(),
                        'school_domain': item['school_domain'].strip().lower(),
                        'is_active': bool(item.get('is_active', True)),
                        'max_uses': int(item.get('max_uses', 100))
                    })
                else:
                    logging.warning(f"Skipping invalid coupon in file: {item}")
            return valid
    except FileNotFoundError:
        return []
    except Exception as e:
        logging.error(f"Failed to load coupons file: {e}")
        return []

def save_coupons_to_file():
    try:
        coupons = CouponCode.query.all()
        payload = [{
            'code': c.code,
            'school_name': c.school_name,
            'school_domain': c.school_domain,
            'is_active': c.is_active,
            'max_uses': c.max_uses
        } for c in coupons]
        with open(app.config['COUPONS_FILE'], 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save coupons file: {e}")

def upsert_coupons_from_file() -> dict:
    """Load coupons.json and upsert into DB by code. Returns counts."""
    file_coupons = load_coupons_from_file()
    created = 0
    updated = 0
    for item in file_coupons:
        code = item['code'].strip().upper()
        existing = CouponCode.query.filter_by(code=code).first()
        if existing:
            # Update existing
            existing.school_name = item['school_name'].strip()
            existing.school_domain = item['school_domain'].strip().lower()
            existing.is_active = bool(item.get('is_active', True))
            existing.max_uses = int(item.get('max_uses', 100))
            updated += 1
        else:
            # Create new
            coupon = CouponCode(
                code=code,
                school_name=item['school_name'].strip(),
                school_domain=item['school_domain'].strip().lower(),
                is_active=bool(item.get('is_active', True)),
                max_uses=int(item.get('max_uses', 100))
            )
            db.session.add(coupon)
            created += 1
    db.session.commit()
    return {'created': created, 'updated': updated}

# Create tables
with app.app_context():
    db.create_all()
    
    # Initialize or update coupons from file
    try:
        stats = upsert_coupons_from_file()
        print(f"SUCCESS: Coupons file sync - created: {stats['created']}, updated: {stats['updated']}")
        logging.info(f"Coupons file sync - created: {stats['created']}, updated: {stats['updated']}")
    except Exception as e:
        print(f"WARNING: Coupon file sync warning: {e}")
        logging.warning(f"Coupon file sync warning: {e}")

# NEW: Calculator utilities (safe evaluator)
import math
import ast

ALLOWED_MATH_FUNCS = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
ALLOWED_CONSTS = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
    'inf': math.inf
}

class SafeEvalVisitor(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call,
        ast.Load, ast.Name, ast.Pow, ast.Mult, ast.Div, ast.Add, ast.Sub,
        ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd
    )

    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError('Disallowed expression')
        return super().visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError('Only simple function names allowed')
        func_name = node.func.id
        if func_name not in ALLOWED_MATH_FUNCS:
            raise ValueError(f'Function {func_name} not allowed')
        for arg in node.args:
            self.visit(arg)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id not in ALLOWED_CONSTS and node.id not in ALLOWED_MATH_FUNCS:
            raise ValueError(f'Name {node.id} not allowed')
        return node

def safe_eval_math(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode='eval')
        SafeEvalVisitor().visit(tree)
        compiled = compile(tree, '<calc>', 'eval')
        return eval(compiled, {'__builtins__': {}}, {**ALLOWED_MATH_FUNCS, **ALLOWED_CONSTS})
    except Exception as e:
        raise ValueError(str(e))

# Lightweight helpers to handle simple math queries without calling the AI
def extract_math_expression(text: str) -> str:
    """Extract a simple arithmetic expression from arbitrary text.
    Supports digits, spaces, + - * / ^ ( ) and decimal points. Also normalizes common symbols.
    Returns an empty string if nothing math-like is found.
    """
    if not text:
        return ""
    # Normalize unicode operators often used by users
    normalized = (
        text.replace('×', '*')
            .replace('x', 'x')  # leave letter x for later filtering
            .replace('X', 'x')
            .replace('÷', '/')
            .replace('^', '**')
    )
    # Strip question words and filler
    normalized = re.sub(r"(?i)^(what\s+is|calculate|compute|solve|evaluate)\s+", "", normalized).strip()
    normalized = normalized.rstrip('?= ').strip()

    # Keep only characters that are valid for our safe evaluator
    allowed_chars_pattern = r"[^0-9\s\.+\-*/()**]"
    candidate = re.sub(allowed_chars_pattern, '', normalized)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # Very basic sanity: must contain at least one digit and one operator
    if re.search(r"\d", candidate) and re.search(r"[+\-*/]", candidate):
        return candidate
    return ""

def try_simple_math_answer(user_text: str) -> str | None:
    """Try to compute a simple math answer. Returns a friendly string or None."""
    expr = extract_math_expression(user_text)
    if not expr:
        return None
    try:
        result = safe_eval_math(expr)
        # Format integers without trailing .0
        if isinstance(result, float) and result.is_integer():
            result_str = str(int(result))
        else:
            result_str = str(result)
        return f"{expr} = {result_str}"
    except Exception:
        return None

def offline_fallback_response(user_text: str) -> str:
    """Return a friendly offline response when AI is unavailable.
    Tries math first, then provides guidance.
    """
    math_answer = try_simple_math_answer(user_text)
    if math_answer:
        return f"Answer: {math_answer}"
    # Lightweight canned help while offline
    return (
        "I'm currently offline and can't access the AI model. "
        "However, you can try rephrasing your question or ask a simple calculation like '2+2'. "
        "Please try again in a bit."
    )

# Helper Functions
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logging.warning("Unauthorized access attempt")
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

def generate_unique_coupon(prefix: str = 'SCH') -> str:
    """Generate a human-friendly unique coupon code."""
    # 12-char code: PREFIX-YYYY-XXXX where XXXX is base36
    random_part = secrets.token_urlsafe(6)  # ~8 chars base64url
    random_part = re.sub(r'[^A-Za-z0-9]', '', random_part)[:6].upper()
    year = datetime.utcnow().year
    return f"{prefix}{year}{random_part}"

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_audio(filename):
    """Check if audio file extension is allowed"""
    ALLOWED_AUDIO_EXTENSIONS = {'webm', 'wav', 'mp3', 'm4a', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file and return the file path"""
    if file and allowed_file(file.filename):
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(file_path)
        logging.info(f"File saved: {file_path}")
        return file_path
    return None

def process_image_with_ocr(image_path):
    """Extract text from image using Gemini Vision API"""
    try:
        if not model:
            return "OCR service is not available"
        
        # Load and prepare image
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        
        # Convert to PIL Image for Gemini
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Prompt specifically for text extraction
        ocr_prompt = """Please extract all the text you can see in this image. 
        
        Instructions:
        - Extract text exactly as it appears
        - Include numbers, symbols, and mathematical expressions
        - Preserve formatting and spacing where possible
        - If there's handwritten text, transcribe it clearly
        - If there's printed text, copy it exactly
        
        Return only the extracted text, nothing else."""
        
        # Generate OCR response
        response = model.generate_content([ocr_prompt, pil_image])
        
        if response and hasattr(response, 'text') and response.text:
            extracted_text = response.text.strip()
            if extracted_text and extracted_text.lower() not in ['no text detected', 'no text found', 'i cannot see any text']:
                return extracted_text
            else:
                return "No text detected in the image"
        else:
            return "No text detected in the image"
        
    except Exception as e:
        logging.error(f"Gemini OCR processing error: {e}")
        return "Error processing image text"

def process_audio_with_gemini(audio_path):
    """Transcribe audio using Gemini if available"""
    try:
        if not model:
            return ""

        ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
        mime_map = {
            'webm': 'audio/webm',
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/m4a',
            'ogg': 'audio/ogg'
        }
        mime_type = mime_map.get(ext, 'audio/webm')

        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        prompt = "Transcribe the following audio to plain text. Return only the transcript."

        response = model.generate_content([
            prompt,
            {
                'mime_type': mime_type,
                'data': audio_bytes
            }
        ])

        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        return ""
    except Exception as e:
        logging.error(f"Gemini audio transcription error: {e}")
        return ""

def process_audio_with_google_speech(audio_path):
    """Transcribe audio using Google Cloud Speech-to-Text if available."""
    try:
        if not speech:
            return None

        client = speech.SpeechClient()

        with open(audio_path, 'rb') as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            # Let the API auto-detect encoding; hint sample rate to improve quality
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code=SPEECH_LANGUAGE,
            enable_automatic_punctuation=True,
            sample_rate_hertz=48000
        )

        response = client.recognize(config=config, audio=audio)
        if not response or not response.results:
            return None

        transcripts = []
        for result in response.results:
            if result.alternatives:
                transcripts.append(result.alternatives[0].transcript)
        text = ' '.join(t.strip() for t in transcripts if t and t.strip()) or None
        # Filter out likely noise like single repeated characters
        if text and len(text) < 3:
            return None
        return text
    except Exception as e:
        logging.error(f"Google Speech-to-Text error: {e}", exc_info=True)
        return None

def analyze_image_with_gemini(image_path, user_query=None, conversation_history=None):
    """Analyze image using Google Gemini Vision with conversation context"""
    try:
        if not model:
            return "AI image analysis is not available"
        
        # Load and prepare image
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        
        # Convert to PIL Image for Gemini
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Prepare prompt with conversation context
        if user_query:
            prompt = f"""You are BYTE, a smart learning assistant. A student has uploaded an image with this question: "{user_query}"

Please analyze the image and help the student. If there are math problems, solve them step by step. If there's text or handwriting, help explain or answer questions about it.

Be educational, clear, and helpful in your response. If this relates to previous conversation, use that context to provide more relevant answers."""
        else:
            prompt = """You are BYTE, a smart learning assistant. A student has uploaded an image. Please analyze what you see in the image and provide helpful educational assistance. 

If you see:
- Math problems: Solve them step by step
- Text or handwriting: Help explain or summarize
- Diagrams: Explain what they show
- Any educational content: Provide relevant help

Be clear, educational, and supportive in your response. If this relates to previous conversation, use that context to provide more relevant answers."""
        
        # Add conversation history to the prompt if available
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\nPrevious conversation context:\n"
            for msg in conversation_history[-6:]:  # Keep last 6 messages for image context
                if msg['sender'] == 'user':
                    history_text += f"Student: {msg['content']}\n"
                else:
                    # Remove HTML tags from bot responses for cleaner context
                    clean_content = re.sub(r'<[^>]+>', '', msg['content'])
                    history_text += f"BYTE: {clean_content}\n"
            
            prompt += history_text
        
        # Generate response
        response = model.generate_content([prompt, pil_image])
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return "I could see your image but couldn't generate a proper analysis. Could you try uploading a clearer image?"
            
    except Exception as e:
        logging.error(f"Gemini image analysis error: {e}")
        return "I had trouble analyzing your image. Please make sure it's a clear image and try again."

# AI Response Functions
def basic_chatbot_responses(user_input):
    """Basic predefined responses for simple queries - EXACT matches only"""
    user_input = user_input.lower().strip()
    
    exact_responses = {
        "hello": "Hello! I'm BYTE, your smart learning assistant. How can I help you today?",
        "hi": "Hi there! I'm here to help you with your learning. What would you like to know?",
        "hey": "Hey! What can I help you learn today?",
        "how are you": "I'm doing great and ready to help you learn! What can I assist you with?",
        "what is your name": "I am BYTE, your AI-powered smart learning assistant created by Team DynamiX.",
        "who are you": "I'm BYTE, designed to help students with their academic questions and learning needs.",
        "what can you do": "I can help you with academic questions, solve math problems, analyze images with text, and much more! Try asking me anything related to your studies.",
        "help": "I'm here to help! You can ask me academic questions, upload images with problems, or even draw problems for me to solve. What do you need help with?",
        "bye": "Goodbye! Feel free to come back anytime you need help with your studies. Happy learning!",
        "goodbye": "See you later! I'm always here when you need academic assistance.",
        "thank you": "You're welcome! I'm glad I could help. Feel free to ask me anything else!",
        "thanks": "Happy to help! Is there anything else you'd like to learn about?"
    }
    
    if user_input in exact_responses:
        logging.info(f"Using basic response for: {user_input}")
        return exact_responses[user_input]
    
    return None

def ask_gemini(query, context=None, conversation_history=None):
    """Enhanced Google Gemini function with conversation memory"""
    try:
        if not model:
            logging.error("Gemini model is not configured")
            return "I'm sorry, but the AI service is not configured properly. Please check the API configuration."
        
        # Enhanced prompt for educational responses
        system_prompt = """You are BYTE, a smart learning assistant created by Team DynamiX to help students with their academic questions.

        Guidelines for responses:
        - Provide clear, educational explanations
        - Include step-by-step solutions for problems
        - Use examples when helpful
        - Keep formatting clean and readable
        - Be encouraging and supportive
        - Focus on helping students understand concepts
        - Keep responses concise but informative
        - Remember previous conversation context and refer to it when relevant
        - If the student asks follow-up questions, use the conversation history to provide context-aware answers

        Student's question: """

        # Build the full prompt with conversation history
        full_prompt = system_prompt + query
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                if msg['sender'] == 'user':
                    history_text += f"Student: {msg['content']}\n"
                else:
                    # Remove HTML tags from bot responses for cleaner context
                    clean_content = re.sub(r'<[^>]+>', '', msg['content'])
                    history_text += f"BYTE: {clean_content}\n"
            
            full_prompt += history_text
        
        if context:
            full_prompt += f"\n\nAdditional context: {context}"

        logging.info(f"Sending query to Gemini with {len(conversation_history) if conversation_history else 0} previous messages")
        
        # Generate response with relaxed safety settings
        response = model.generate_content(
            full_prompt,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        if response and hasattr(response, 'text') and response.text:
            answer = response.text.strip()
            logging.info(f"Received Gemini response: {len(answer)} characters")
            return answer
        else:
            logging.warning("Empty or invalid response from Gemini")
            return "I couldn't generate a response to that question. Could you try rephrasing it?"
            
    except Exception as e:
        logging.error(f"Gemini API Error: {str(e)}", exc_info=True)
        return f"I'm experiencing some technical difficulties right now. Please try again in a moment."

def format_response_as_html(response_text):
    """Format AI response as HTML for better presentation"""
    if not response_text:
        return response_text
    
    # Simple formatting to make responses more readable
    formatted_text = response_text
    
    # Convert **text** to bold
    formatted_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted_text)
    
    # Convert *text* to italic  
    formatted_text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', formatted_text)
    
    # Convert markdown headers
    formatted_text = re.sub(r'^#{1,3}\s+(.+)$', r'<h3 style="color:#333; margin-top:15px; margin-bottom:8px; font-weight:600;">\1</h3>', formatted_text, flags=re.MULTILINE)
    
    # Convert bullet points
    formatted_text = re.sub(r'^\*\s+', '• ', formatted_text, flags=re.MULTILINE)
    
    # Split into paragraphs and format
    lines = formatted_text.split('\n')
    formatted_response = "<div style='font-family:Arial, sans-serif; line-height:1.6; font-size:16px;'>"

    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped:
            if '<h3>' in line_stripped:
                formatted_response += line_stripped
            elif re.match(r'^\d+\.\s+', line_stripped):
                formatted_response += f"<p style='margin-left:20px; margin-bottom:8px;'><strong>{line_stripped}</strong></p>"
            elif line_stripped.startswith('• '):
                content = line_stripped[2:]
                formatted_response += f"<p style='margin-left:20px; margin-bottom:8px;'>• {content}</p>"
            else:
                formatted_response += f"<p style='margin-bottom:12px;'>{line_stripped}</p>"
        else:
            formatted_response += "<br>"

    formatted_response += "</div>"
    return formatted_response

# Static file routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Note: The generic static file route must be placed AFTER API routes to avoid shadowing them.

# NEW: Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Authentication Routes (same as before)
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        logging.debug(f"Signup request received: {request.method}")
        data = request.get_json()
        logging.debug(f"Signup data: {data}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        login_method = data.get('login_method', 'form')
        coupon_code = data.get('coupon_code', '').strip().upper()
        
        if not name or not email:
            return jsonify({'error': 'Name and email are required'}), 400
        
        if login_method == 'form' and not password:
            return jsonify({'error': 'Password is required'}), 400
        
        if login_method == 'coupon' and not coupon_code:
            return jsonify({'error': 'Coupon code is required'}), 400
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'User with this email already exists'}), 400
        
        # Handle coupon-based signup
        if login_method == 'coupon':
            coupon = CouponCode.query.filter_by(code=coupon_code).first()
            if not coupon:
                return jsonify({'error': 'Invalid coupon code'}), 401
            
            if not coupon.is_valid():
                return jsonify({'error': 'Coupon code is expired or no longer valid'}), 401
            
            if not coupon.can_use_for_email(email):
                return jsonify({'error': f'Coupon code is not valid for {email.split("@")[1]} domain'}), 401
            
            user = User(
                name=name, 
                email=email, 
                login_method='coupon',
                school_id=coupon.school_name
            )
            
            # Increment coupon usage
            coupon.current_uses += 1
            save_coupons_to_file()
        else:
            user = User(name=name, email=email, login_method=login_method)
            if password:
                user.set_password(password)
        
        db.session.add(user)
        db.session.commit()

        # If signup via school coupon, generate a unique user coupon
        if user.login_method == 'coupon':
            existing_uc = UserCoupon.query.filter_by(user_id=user.id).first()
            if not existing_uc:
                # Ensure code uniqueness
                for _ in range(5):
                    code = generate_unique_coupon()
                    if not UserCoupon.query.filter_by(code=code).first():
                        uc = UserCoupon(user_id=user.id, code=code, school_name=user.school_id)
                        db.session.add(uc)
                        db.session.commit()
                        break
        
        session['user_id'] = user.id
        session.permanent = True
        session.modified = True
        
        logging.info(f"New user registered: {email}")
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user': user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Signup error: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        logging.debug(f"Login request received: {request.method}")
        data = request.get_json()
        logging.debug(f"Login data keys: {list(data.keys()) if data else 'No data'}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        login_method = data.get('login_method', 'form')
        coupon_code = data.get('coupon_code', '').strip().upper()
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if login_method == 'form':
            if not user or not user.check_password(password):
                return jsonify({'error': 'Invalid email or password'}), 401
        elif login_method == 'coupon':
            # Handle coupon-based login
            if not coupon_code:
                return jsonify({'error': 'Coupon code is required'}), 400
            
            # Validate coupon code
            coupon = CouponCode.query.filter_by(code=coupon_code).first()
            if not coupon:
                return jsonify({'error': 'Invalid coupon code'}), 401
            
            if not coupon.is_valid():
                return jsonify({'error': 'Coupon code is expired or no longer valid'}), 401
            
            if not coupon.can_use_for_email(email):
                return jsonify({'error': f'Coupon code is not valid for {email.split("@")[1]} domain'}), 401
            
            # Create or update user
            if not user:
                name = data.get('name', email.split('@')[0])
                user = User(
                    name=name, 
                    email=email, 
                    login_method='coupon',
                    school_id=coupon.school_name
                )
                db.session.add(user)
            else:
                # Update existing user's login method and school ID
                user.login_method = 'coupon'
                user.school_id = coupon.school_name
            
            # Increment coupon usage
            coupon.current_uses += 1
            db.session.commit()
            save_coupons_to_file()

            # Ensure the user has a unique coupon code stored
            existing_uc = UserCoupon.query.filter_by(user_id=user.id).first()
            if not existing_uc:
                for _ in range(5):
                    code = generate_unique_coupon()
                    if not UserCoupon.query.filter_by(code=code).first():
                        uc = UserCoupon(user_id=user.id, code=code, school_name=user.school_id)
                        db.session.add(uc)
                        db.session.commit()
                        break
            
            logging.info(f"Coupon login successful: {email} using {coupon_code}")
        else:
            if not user:
                name = data.get('name', email.split('@')[0])
                user = User(name=name, email=email, login_method=login_method)
                db.session.add(user)
                db.session.commit()
        
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        session['user_id'] = user.id
        session.permanent = True
        session.modified = True
        
        logging.info(f"User logged in: {email}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Login error: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    user = get_current_user()
    if user:
        logging.info(f"User logged out: {user.email}")
    session.clear()
    return jsonify({'success': True, 'message': 'Logout successful'})

@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user_info():
    user = get_current_user()
    if user:
        # Attach user coupon if any
        uc = UserCoupon.query.filter_by(user_id=user.id).first()
        user_data = user.to_dict()
        if uc:
            user_data['user_coupon'] = uc.to_dict()
        return jsonify({'user': user_data})
    return jsonify({'error': 'User not found'}), 404

# Google OAuth Routes
@app.route('/api/auth/google')
def google_login():
    """Initiate Google OAuth login"""
    try:
        # Generate state parameter for security
        state = secrets.token_urlsafe(32)
        
        # Make session permanent and save state
        session.permanent = True
        session['oauth_state'] = state
        session['oauth_timestamp'] = datetime.utcnow().timestamp()
        
        # Force session save
        session.modified = True
        
        # Build Google OAuth URL with dynamic redirect URI matching current host
        redirect_uri = get_google_redirect_uri()
        logging.info(f"Google OAuth using redirect_uri: {redirect_uri}")
        params = {
            'client_id': GOOGLE_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',
            'prompt': 'consent'
        }

        google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        
        logging.info(f"Redirecting to Google OAuth with state: {state}")
        return redirect(google_auth_url)
        
    except Exception as e:
        logging.error(f"Google OAuth initiation error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to initiate Google login'}), 500

# Debug endpoint to display the exact redirect_uri the app is sending to Google
@app.route('/api/auth/google/redirect_uri')
def google_redirect_uri_debug():
    try:
        uri = get_google_redirect_uri()
        return jsonify({
            'redirect_uri': uri,
            'client_id': GOOGLE_CLIENT_ID,
            'note': 'Add this exact redirect_uri to Google Cloud Console → Credentials → OAuth 2.0 Client → Authorized redirect URIs.'
        })
    except Exception as e:
        logging.error(f"Redirect URI debug error: {e}")
        return jsonify({'error': 'Failed to compute redirect_uri'}), 500

@app.route('/api/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback"""
    try:
        # Verify state parameter
        state = request.args.get('state')
        stored_state = session.get('oauth_state')
        oauth_timestamp = session.get('oauth_timestamp', 0)
        
        # Check if state exists and is recent (within 10 minutes)
        current_time = datetime.utcnow().timestamp()
        if not state or not stored_state or state != stored_state:
            logging.error(f"Invalid OAuth state parameter. Expected: {stored_state}, Got: {state}")
            return redirect('/auth.html?google_error=true&reason=invalid_state')
        
        # Check if OAuth request is too old (20 minutes)
        if current_time - oauth_timestamp > 1200:  # 20 minutes window
            logging.error("OAuth state expired")
            return redirect('/auth.html?google_error=true&reason=expired')
        
        # Clear OAuth data from session
        session.pop('oauth_state', None)
        session.pop('oauth_timestamp', None)
        
        # Get authorization code
        code = request.args.get('code')
        if not code:
            logging.error("No authorization code received")
            return jsonify({'error': 'No authorization code received'}), 400
        
        # Exchange code for access token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': get_google_redirect_uri()
        }
        
        logging.info(f"Exchanging code with redirect_uri: {token_data.get('redirect_uri')}")
        token_response = requests.post(token_url, data=token_data)
        if not token_response.ok:
            logging.error(f"Token exchange failed: {token_response.text}")
            return jsonify({'error': 'Failed to exchange authorization code'}), 500
        
        token_info = token_response.json()
        access_token = token_info.get('access_token')
        
        if not access_token:
            logging.error("No access token received")
            return jsonify({'error': 'No access token received'}), 500
        
        # Get user info from Google
        user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {'Authorization': f'Bearer {access_token}'}
        user_response = requests.get(user_info_url, headers=headers)
        
        if not user_response.ok:
            logging.error(f"Failed to get user info: {user_response.text}")
            return jsonify({'error': 'Failed to get user information'}), 500
        
        google_user = user_response.json()
        
        # Extract user information
        email = google_user.get('email', '').lower()
        name = google_user.get('name', '')
        google_id = google_user.get('id', '')
        
        if not email:
            logging.error("No email received from Google")
            return jsonify({'error': 'No email received from Google'}), 500
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Update existing user's login method if needed
            if user.login_method != 'google':
                user.login_method = 'google'
                db.session.commit()
        else:
            # Create new user
            user = User(
                name=name or email.split('@')[0],
                email=email,
                login_method='google'
            )
            db.session.add(user)
            db.session.commit()
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Set session
        session['user_id'] = user.id
        session.permanent = True
        session.modified = True
        
        logging.info(f"Google OAuth login successful for: {email}")
        
        # Redirect to chatbot page after successful login
        return redirect('/chatbot.html')
        
    except Exception as e:
        logging.error(f"Google OAuth callback error: {e}", exc_info=True)
        return redirect('/auth.html?google_error=true')

# Chat Routes
@app.route('/api/chats', methods=['GET'])
@login_required
def get_chats():
    try:
        user = get_current_user()
        chats = Chat.query.filter_by(user_id=user.id).order_by(Chat.updated_at.desc()).all()
        
        chat_data = []
        for chat in chats:
            chat_dict = chat.to_dict()
            first_message = Message.query.filter_by(
                chat_id=chat.id, 
                sender='user'
            ).order_by(Message.timestamp.asc()).first()
            
            if first_message:
                if not chat.title:
                    chat_dict['title'] = first_message.content[:50] + ('...' if len(first_message.content) > 50 else '')
                chat_dict['preview'] = first_message.content[:100] + ('...' if len(first_message.content) > 100 else '')
            else:
                chat_dict['preview'] = 'No messages yet'
            
            chat_data.append(chat_dict)
        
        return jsonify({'chats': chat_data})
        
    except Exception as e:
        logging.error(f"Error getting chats: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats', methods=['POST'])
@login_required
def create_chat():
    try:
        user = get_current_user()
        data = request.get_json() or {}
        
        title = data.get('title', None)
        chat = Chat(user_id=user.id, title=title)
        
        db.session.add(chat)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'chat': chat.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating chat: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats/<int:chat_id>', methods=['GET'])
@login_required
def get_chat(chat_id):
    try:
        user = get_current_user()
        chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp.asc()).all()
        
        return jsonify({
            'chat': chat.to_dict(),
            'messages': [msg.to_dict() for msg in messages]
        })
        
    except Exception as e:
        logging.error(f"Error getting chat {chat_id}: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    try:
        user = get_current_user()
        chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Delete associated files
        messages_with_files = Message.query.filter_by(chat_id=chat_id).filter(Message.file_path.isnot(None)).all()
        for message in messages_with_files:
            try:
                if os.path.exists(message.file_path):
                    os.remove(message.file_path)
                    logging.info(f"Deleted file: {message.file_path}")
            except Exception as e:
                logging.warning(f"Failed to delete file {message.file_path}: {e}")
        
        # Delete chat (messages will be deleted automatically due to cascade)
        db.session.delete(chat)
        db.session.commit()
        
        logging.info(f"Chat {chat_id} deleted by user {user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Chat deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting chat {chat_id}: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats/<int:chat_id>/clear', methods=['POST'])
@login_required
def clear_chat(chat_id):
    try:
        user = get_current_user()
        chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Delete associated files
        messages_with_files = Message.query.filter_by(chat_id=chat_id).filter(Message.file_path.isnot(None)).all()
        for message in messages_with_files:
            try:
                if os.path.exists(message.file_path):
                    os.remove(message.file_path)
                    logging.info(f"Deleted file: {message.file_path}")
            except Exception as e:
                logging.warning(f"Failed to delete file {message.file_path}: {e}")
        
        # Delete all messages in the chat
        Message.query.filter_by(chat_id=chat_id).delete()
        
        # Update chat timestamp
        chat.updated_at = datetime.utcnow()
        db.session.commit()
        
        logging.info(f"Chat {chat_id} cleared by user {user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Chat cleared successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error clearing chat {chat_id}: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chats/<int:chat_id>/rename', methods=['PUT'])
@login_required
def rename_chat(chat_id):
    try:
        user = get_current_user()
        data = request.get_json()
        
        if not data or 'title' not in data:
            return jsonify({'error': 'Title is required'}), 400
        
        new_title = data['title'].strip()
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400
        
        if len(new_title) > 200:
            return jsonify({'error': 'Title too long (max 200 characters)'}), 400
        
        chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
        
        if not chat:
            return jsonify({'error': 'Chat not found'}), 404
        
        chat.title = new_title
        chat.updated_at = datetime.utcnow()
        db.session.commit()
        
        logging.info(f"Chat {chat_id} renamed to '{new_title}' by user {user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Chat renamed successfully',
            'chat': chat.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error renaming chat {chat_id}: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# MAIN ASK ROUTE
@app.route('/api/ask', methods=['POST'])
@login_required
def ask_question():
    try:
        logging.info("=== ASK QUESTION START ===")
        user = get_current_user()
        logging.info(f"User: {user.email}")
        
        # Debug incoming request
        logging.info(f"Request content type: {request.content_type}")
        logging.info(f"Request form keys: {list(request.form.keys())}")
        logging.info(f"Request files keys: {list(request.files.keys())}")
        
        # Get chat ID
        chat_id = request.form.get('chat_id')
        logging.info(f"Chat ID from request: {chat_id}")
        
        if chat_id:
            try:
                chat_id = int(chat_id)
                chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
                if not chat:
                    logging.error(f"Chat {chat_id} not found for user {user.id}")
                    return jsonify({'error': 'Chat not found'}), 404
                logging.info(f"Using existing chat: {chat_id}")
            except ValueError:
                logging.error(f"Invalid chat_id: {chat_id}")
                return jsonify({'error': 'Invalid chat ID'}), 400
        else:
            # Create new chat
            chat = Chat(user_id=user.id)
            db.session.add(chat)
            db.session.flush()  # Get the ID without committing
            logging.info(f"Created new chat: {chat.id}")
        
        # Get user query
        query = request.form.get('query', '').strip()
        logging.info(f"Query: '{query}'")
        
        if not query:
            logging.error("No query provided")
            return jsonify({'error': 'Query is required'}), 400
        
        # Save user message
        user_message = Message(
            chat_id=chat.id,
            sender='user',
            content=query,
            message_type='text'
        )
        db.session.add(user_message)
        
        # Generate AI response
        logging.info("=== GENERATING AI RESPONSE ===")
        
        # Get conversation history for context
        conversation_history = []
        if chat.id:
            # Get previous messages in this chat (excluding the current user message)
            previous_messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.timestamp.asc()).all()
            conversation_history = [msg.to_dict() for msg in previous_messages]
            logging.info(f"Retrieved {len(conversation_history)} previous messages for context")
        
        # Try basic responses first
        ai_response = basic_chatbot_responses(query)
        if ai_response:
            logging.info("Using basic response")
        else:
            # Try quick math fallback before calling the AI
            math_answer = try_simple_math_answer(query)
            if math_answer:
                ai_response = f"Answer: {math_answer}"
                logging.info("Using simple math evaluator response")
            else:
                logging.info("Using Gemini for response with conversation history")
                ai_response = ask_gemini(query, conversation_history=conversation_history)
                # If Gemini returns the generic technical difficulties message, replace with friendlier offline fallback
                if ai_response and "technical difficulties" in ai_response.lower():
                    logging.warning("Gemini returned technical difficulties; using offline fallback")
                    ai_response = offline_fallback_response(query)
        
        # Validate response
        if not ai_response or ai_response.strip() == "":
            ai_response = "I apologize, but I couldn't generate a response to your question. Please try rephrasing it or ask something else."
            logging.warning("Generated empty response, using fallback")
        
        logging.info(f"Final AI response length: {len(ai_response)} characters")
        logging.debug(f"AI response preview: {ai_response[:200]}...")
        
        # Format response
        formatted_response = format_response_as_html(ai_response)
        
        # Save AI message
        ai_message = Message(
            chat_id=chat.id,
            sender='bot',
            content=formatted_response,
            message_type='text'
        )
        db.session.add(ai_message)
        
        # Update chat title if needed
        if not chat.title and len(query.split()) > 0:
            title_words = query.split()[:8]
            chat.title = ' '.join(title_words) + ('...' if len(query.split()) > 8 else '')
            logging.info(f"Set chat title: {chat.title}")
        
        chat.updated_at = datetime.utcnow()
        db.session.commit()
        
        logging.info("=== ASK QUESTION SUCCESS ===")
        
        return jsonify({
            'response': formatted_response,
            'chat_id': chat.id,
            'message_id': ai_message.id,
            'chat_title': chat.title
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error in ask_question: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NEW: IMAGE UPLOAD AND HANDWRITING RECOGNITION ROUTE
@app.route('/api/recognize_handwriting', methods=['POST'])
@login_required
def recognize_handwriting():
    try:
        logging.info("=== IMAGE RECOGNITION START ===")
        user = get_current_user()
        logging.info(f"User: {user.email}")
        
        # Debug incoming request
        logging.info(f"Request content type: {request.content_type}")
        logging.info(f"Request form keys: {list(request.form.keys())}")
        logging.info(f"Request files keys: {list(request.files.keys())}")
        
        # Check if image file is present
        if 'image' not in request.files:
            logging.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logging.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WEBP)'}), 400
        
        # Get chat ID
        chat_id = request.form.get('chat_id')
        logging.info(f"Chat ID from request: {chat_id}")
        
        if chat_id:
            try:
                chat_id = int(chat_id)
                chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()
                if not chat:
                    logging.error(f"Chat {chat_id} not found for user {user.id}")
                    return jsonify({'error': 'Chat not found'}), 404
                logging.info(f"Using existing chat: {chat_id}")
            except ValueError:
                logging.error(f"Invalid chat_id: {chat_id}")
                return jsonify({'error': 'Invalid chat ID'}), 400
        else:
            # Create new chat
            chat = Chat(user_id=user.id)
            db.session.add(chat)
            db.session.flush()  # Get the ID without committing
            logging.info(f"Created new chat: {chat.id}")
        
        # Save the uploaded file
        file_path = save_uploaded_file(file)
        if not file_path:
            logging.error("Failed to save uploaded file")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        logging.info(f"File saved to: {file_path}")
        
        # Get user query if provided
        user_query = request.form.get('query', '').strip()
        logging.info(f"User query: '{user_query}'")
        
        # Save user's image message
        # Create URL for the uploaded image
        image_filename = os.path.basename(file_path)
        image_url = f"/uploads/{image_filename}"
        
        # Save image message
        image_message = Message(
            chat_id=chat.id,
            sender='user',
            content=image_url,
            message_type='image',
            file_path=file_path
        )
        db.session.add(image_message)
        
        # If user provided a text query along with image, save that too
        if user_query:
            text_message = Message(
                chat_id=chat.id,
                sender='user',
                content=user_query,
                message_type='text'
            )
            db.session.add(text_message)
        
        # Process the image
        logging.info("=== PROCESSING IMAGE ===")
        
        # Get conversation history for context
        conversation_history = []
        if chat.id:
            # Get previous messages in this chat (excluding the current user message)
            previous_messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.timestamp.asc()).all()
            conversation_history = [msg.to_dict() for msg in previous_messages]
            logging.info(f"Retrieved {len(conversation_history)} previous messages for image context")
        
        # First, try OCR to extract text
        extracted_text = process_image_with_ocr(file_path)
        logging.info(f"OCR extracted text: {extracted_text[:100]}...")
        
        # Then analyze with Gemini Vision
        ai_response = analyze_image_with_gemini(file_path, user_query, conversation_history)
        
        # If OCR found text and Gemini response is basic, enhance it
        if extracted_text and "No text detected" not in extracted_text and "Error processing" not in extracted_text:
            if len(ai_response) < 100:  # If Gemini response is too short
                # Combine OCR results with Gemini analysis
                enhanced_query = f"The image contains this text: '{extracted_text}'. "
                if user_query:
                    enhanced_query += f"The user asked: '{user_query}'. "
                enhanced_query += "Please help analyze and explain this content."
                
                enhanced_response = ask_gemini(enhanced_query, conversation_history=conversation_history)
                if enhanced_response and len(enhanced_response) > len(ai_response):
                    ai_response = enhanced_response
        
        # Validate response
        if not ai_response or ai_response.strip() == "":
            ai_response = "I could see your image but had trouble analyzing it. Could you try uploading a clearer image or ask a specific question about it?"
            logging.warning("Generated empty response, using fallback")
        
        logging.info(f"Final AI response length: {len(ai_response)} characters")
        logging.debug(f"AI response preview: {ai_response[:200]}...")
        
        # Format response
        formatted_response = format_response_as_html(ai_response)
        
        # Save AI message
        ai_message = Message(
            chat_id=chat.id,
            sender='bot',
            content=formatted_response,
            message_type='text'
        )
        db.session.add(ai_message)
        
        # Update chat title if needed
        if not chat.title:
            if user_query:
                title_words = user_query.split()[:6]
                chat.title = ' '.join(title_words) + ('...' if len(user_query.split()) > 6 else '')
            elif extracted_text and "No text detected" not in extracted_text:
                title_words = extracted_text.split()[:6]
                chat.title = ' '.join(title_words) + ('...' if len(extracted_text.split()) > 6 else '')
            else:
                chat.title = "Image Analysis"
            logging.info(f"Set chat title: {chat.title}")
        
        chat.updated_at = datetime.utcnow()
        db.session.commit()
        
        logging.info("=== IMAGE RECOGNITION SUCCESS ===")
        
        return jsonify({
            'response': formatted_response,
            'chat_id': chat.id,
            'message_id': ai_message.id,
            'chat_title': chat.title,
            'extracted_text': extracted_text if "Error processing" not in extracted_text else None,
            'image_url': image_url
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error in recognize_handwriting: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NEW: Upload general files route
@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        logging.info("=== FILE UPLOAD START ===")
        user = get_current_user()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type (images only)
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WEBP)'}), 400
        
        # Save file
        file_path = save_uploaded_file(file)
        if not file_path:
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Create URL for the uploaded image
        image_filename = os.path.basename(file_path)
        image_url = f"/uploads/{image_filename}"
        
        logging.info(f"File uploaded successfully: {file_path}")
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'file_path': file_path,
            'image_url': image_url,
            'filename': file.filename
        })
        
    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# NEW: Voice to text transcription route
@app.route('/api/voice_to_text', methods=['POST'])
@login_required
def voice_to_text():
    try:
        logging.info("=== VOICE TRANSCRIPTION START ===")
        user = get_current_user()
        logging.info(f"User: {user.email}")

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_audio(file.filename):
            return jsonify({'error': 'Invalid audio type. Please upload WEBM, WAV, MP3, M4A or OGG'}), 400

        # Save uploaded audio
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        logging.info(f"Audio saved: {file_path}")

        # Prefer Google Speech-to-Text, fallback to Gemini
        transcript = process_audio_with_google_speech(file_path)
        if not transcript:
            transcript = process_audio_with_gemini(file_path)
        if not transcript:
            transcript = ""

        return jsonify({
            'success': True,
            'transcript': transcript,
            'audio_url': f"/uploads/{filename}"
        })
    except Exception as e:
        logging.error(f"Error in voice_to_text: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Test endpoint to check if API is working
@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    return jsonify({
        'status': 'API is working',
        'method': request.method,
        'timestamp': datetime.utcnow().isoformat(),
        'gemini_available': model is not None,
        'ocr_available': ocr_reader == "gemini"
    })

# NEW: Calculator API
@app.route('/api/calc', methods=['POST'])
@login_required
def calculator_api():
    try:
        data = request.get_json() or {}
        expr = (data.get('expression') or '').strip()
        if not expr:
            return jsonify({'error': 'expression is required'}), 400
        result = safe_eval_math(expr)
        return jsonify({'success': True, 'result': result})
    except ValueError as ve:
        return jsonify({'error': f'Invalid expression: {ve}'}), 400
    except Exception as e:
        logging.error(f"Calculator error: {e}", exc_info=True)
        return jsonify({'error': 'Calculation failed'}), 500

# NEW: Tasks CRUD API
@app.route('/api/tasks', methods=['GET', 'POST'])
@login_required
def tasks_collection():
    try:
        user = get_current_user()
        if request.method == 'GET':
            tasks = Task.query.filter_by(user_id=user.id).order_by(Task.created_at.desc()).all()
            return jsonify({'tasks': [t.to_dict() for t in tasks]})
        data = request.get_json() or {}
        content = (data.get('content') or '').strip()
        if not content:
            return jsonify({'error': 'content is required'}), 400
        task = Task(user_id=user.id, content=content, is_completed=bool(data.get('is_completed', False)))
        db.session.add(task)
        db.session.commit()
        return jsonify({'success': True, 'task': task.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        logging.error(f"Tasks collection error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process tasks request'}), 500

@app.route('/api/tasks/<int:task_id>', methods=['PUT', 'DELETE'])
@login_required
def task_item(task_id):
    try:
        user = get_current_user()
        task = Task.query.filter_by(id=task_id, user_id=user.id).first()
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        if request.method == 'DELETE':
            db.session.delete(task)
            db.session.commit()
            return jsonify({'success': True})
        # PUT
        data = request.get_json() or {}
        if 'content' in data:
            new_content = (data.get('content') or '').strip()
            if not new_content:
                return jsonify({'error': 'content cannot be empty'}), 400
            task.content = new_content
        if 'is_completed' in data:
            task.is_completed = bool(data.get('is_completed'))
        db.session.commit()
        return jsonify({'success': True, 'task': task.to_dict()})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Task item error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process task'}), 500

# Debug endpoint for session troubleshooting
@app.route('/api/debug/session', methods=['GET'])
def debug_session():
    """Debug endpoint to check session state"""
    return jsonify({
        'session_id': session.get('_id'),
        'user_id': session.get('user_id'),
        'oauth_state': session.get('oauth_state'),
        'oauth_timestamp': session.get('oauth_timestamp'),
        'session_permanent': session.permanent,
        'session_modified': session.modified,
        'cookies': dict(request.cookies)
    })

# Coupon management endpoints
@app.route('/api/admin/coupons', methods=['GET'])
def get_coupons():
    """Get all coupon codes (admin endpoint)"""
    try:
        coupons = CouponCode.query.all()
        return jsonify({
            'coupons': [{
                'id': c.id,
                'code': c.code,
                'school_name': c.school_name,
                'school_domain': c.school_domain,
                'is_active': c.is_active,
                'max_uses': c.max_uses,
                'current_uses': c.current_uses,
                'created_at': c.created_at.isoformat(),
                'expires_at': c.expires_at.isoformat() if c.expires_at else None,
                'is_valid': c.is_valid()
            } for c in coupons]
        })
    except Exception as e:
        logging.error(f"Error getting coupons: {e}")
        return jsonify({'error': 'Failed to get coupons'}), 500

@app.route('/api/admin/coupons/export', methods=['GET'])
def export_coupons():
    """Export coupons to JSON file and return payload."""
    try:
        save_coupons_to_file()
        with open(app.config['COUPONS_FILE'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'coupons': data})
    except Exception as e:
        logging.error(f"Error exporting coupons: {e}")
        return jsonify({'error': 'Failed to export coupons'}), 500

@app.route('/api/user/coupon', methods=['GET', 'POST'])
@login_required
def user_coupon():
    """Get or create the current user's unique coupon (for school-id logins)."""
    try:
        user = get_current_user()
        if request.method == 'GET':
            uc = UserCoupon.query.filter_by(user_id=user.id).first()
            if not uc:
                return jsonify({'coupon': None})
            return jsonify({'coupon': uc.to_dict()})

        # POST: create/regenerate if none exists; does not overwrite existing
        uc = UserCoupon.query.filter_by(user_id=user.id).first()
        if uc:
            return jsonify({'coupon': uc.to_dict(), 'existing': True})

        if user.login_method != 'coupon':
            return jsonify({'error': 'Coupon generation allowed only for school ID users'}), 400

        for _ in range(5):
            code = generate_unique_coupon()
            if not UserCoupon.query.filter_by(code=code).first():
                uc = UserCoupon(user_id=user.id, code=code, school_name=user.school_id)
                db.session.add(uc)
                db.session.commit()
                return jsonify({'coupon': uc.to_dict(), 'existing': False})

        return jsonify({'error': 'Failed to generate unique coupon'}), 500
    except Exception as e:
        logging.error(f"Error in user_coupon: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to get/create coupon'}), 500

@app.route('/api/admin/coupons', methods=['POST'])
def create_coupon():
    """Create a new coupon code (admin endpoint)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        ok, reason = validate_coupon_data(data)
        if not ok:
            return jsonify({'error': reason}), 400
        
        # Check if coupon code already exists
        existing = CouponCode.query.filter_by(code=data['code'].strip().upper()).first()
        if existing:
            return jsonify({'error': 'Coupon code already exists'}), 400
        
        coupon = CouponCode(
            code=data['code'].strip().upper(),
            school_name=data['school_name'].strip(),
            school_domain=data['school_domain'].strip().lower(),
            max_uses=int(data.get('max_uses', 100)),
            is_active=bool(data.get('is_active', True))
        )
        
        # Set expiration if provided
        if data.get('expires_at'):
            try:
                coupon.expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid expiration date format'}), 400
        
        db.session.add(coupon)
        db.session.commit()
        # Persist to file store
        save_coupons_to_file()
        
        logging.info(f"New coupon created: {coupon.code} for {coupon.school_name}")
        
        return jsonify({
            'success': True,
            'message': 'Coupon created successfully',
            'coupon': {
                'id': coupon.id,
                'code': coupon.code,
                'school_name': coupon.school_name,
                'school_domain': coupon.school_domain,
                'is_active': coupon.is_active,
                'max_uses': coupon.max_uses,
                'current_uses': coupon.current_uses
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating coupon: {e}")
        return jsonify({'error': 'Failed to create coupon'}), 500

@app.route('/api/admin/coupons/<int:coupon_id>', methods=['PUT'])
def update_coupon(coupon_id):
    """Update a coupon code (admin endpoint)"""
    try:
        coupon = CouponCode.query.get(coupon_id)
        if not coupon:
            return jsonify({'error': 'Coupon not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update fields
        if 'school_name' in data:
            coupon.school_name = data['school_name']
        if 'school_domain' in data:
            coupon.school_domain = data['school_domain']
        if 'is_active' in data:
            coupon.is_active = data['is_active']
        if 'max_uses' in data:
            coupon.max_uses = data['max_uses']
        if 'expires_at' in data:
            if data['expires_at']:
                try:
                    coupon.expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
                except ValueError:
                    return jsonify({'error': 'Invalid expiration date format'}), 400
            else:
                coupon.expires_at = None
        
        db.session.commit()
        save_coupons_to_file()
        
        logging.info(f"Coupon updated: {coupon.code}")
        
        return jsonify({
            'success': True,
            'message': 'Coupon updated successfully',
            'coupon': {
                'id': coupon.id,
                'code': coupon.code,
                'school_name': coupon.school_name,
                'school_domain': coupon.school_domain,
                'is_active': coupon.is_active,
                'max_uses': coupon.max_uses,
                'current_uses': coupon.current_uses,
                'expires_at': coupon.expires_at.isoformat() if coupon.expires_at else None
            }
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating coupon: {e}")
        return jsonify({'error': 'Failed to update coupon'}), 500

@app.route('/api/admin/coupons/<int:coupon_id>', methods=['DELETE'])
def delete_coupon(coupon_id):
    """Delete a coupon code (admin endpoint)"""
    try:
        coupon = CouponCode.query.get(coupon_id)
        if not coupon:
            return jsonify({'error': 'Coupon not found'}), 404
        
        db.session.delete(coupon)
        db.session.commit()
        save_coupons_to_file()
        
        logging.info(f"Coupon deleted: {coupon.code}")
        
        return jsonify({
            'success': True,
            'message': 'Coupon deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting coupon: {e}")
        return jsonify({'error': 'Failed to delete coupon'}), 500

# Health check with more details
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        db.session.execute(db.text('SELECT 1'))
        db_status = True
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        db_status = False
    
    # Test Gemini
    gemini_status = False
    if model:
        try:
            test_response = model.generate_content("Say 'Hello, this is a test'")
            if test_response and hasattr(test_response, 'text'):
                gemini_status = True
        except Exception as e:
            logging.error(f"Gemini health check failed: {e}")
    
    # Test OCR
    ocr_status = ocr_reader == "gemini"
    
    return jsonify({
        'status': 'healthy' if (db_status and gemini_status) else 'partial' if db_status else 'unhealthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected' if db_status else 'disconnected',
        'gemini': 'working' if gemini_status else 'not working',
        'ocr': 'available' if ocr_status else 'not available',
        'version': '2.4.0'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logging.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

# Generic static file route (placed after API routes to prevent shadowing)
@app.route('/<path:filename>')
def serve_static(filename):
    
    if filename.startswith('api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return send_from_directory('.', filename)


@app.route('/todo.html')
def serve_todo_page():
    return send_from_directory('.', 'todo.html')

@app.route('/calculator.html')
def serve_calculator_page():
    return send_from_directory('.', 'calculator.html')

@app.route('/clock.html')
def serve_clock_page():
    return send_from_directory('.', 'clock.html')

if __name__ == '__main__':     
    print("STARTING: BYTE Server starting...")
    print("INFO: Server running at: http://127.0.0.1:5000/")
    try:         
        app.run(host='0.0.0.0', port=5000)  
    except Exception as e:         
        print(f"ERROR: Server error: {e}")
        logging.error(f"Server error: {e}")     
    finally:         
        print("BYTE Server shutdown complete")
        logging.info("BYTE Server shutdown complete")
