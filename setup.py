# setup.py - Run this to install required packages
import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    packages = [
        'flask',
        'flask-sqlalchemy', 
        'werkzeug',
        'pillow',
        'requests',
        'python-dotenv',
        'google-generativeai',
        'google-cloud-speech'
    ]
    
    print("🔧 Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def create_folders():
    """Create necessary folders"""
    folders = ['uploads', 'instance']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Created {folder} folder")

def setup_env_file():
    """Create .env file template"""
    env_content = """# BYTE Environment Variables
# Get your Google Gemini API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Set Flask environment
FLASK_ENV=development
FLASK_DEBUG=True
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("📄 Created .env file template")
        print("⚠️  Please add your Google Gemini API key to the .env file")
    else:
        print("✅ .env file already exists")

if __name__ == "__main__":
    print("🚀 Setting up BYTE application...")
    
    install_packages()
    create_folders()
    setup_env_file()
    
    print("\n" + "="*50)
    print("🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Get a Google Gemini API key from: https://aistudio.google.com/app/apikey")
    print("2. Add your API key to backend.py (replace 'your_gemini_api_key_here')")
    print("3. Run: python backend.py")
    print("4. Open your browser to: http://localhost:5000")
    print("\n💡 Note: Google Gemini API has a free tier with generous limits!")
    print("💰 Pricing info: https://ai.google.dev/pricing")
    print("📖 Documentation: https://ai.google.dev/docs")
    print("="*50)