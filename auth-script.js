// Navigation functions
function goToHomepage() {
    const backBtn = document.querySelector('.back-btn');
    backBtn.classList.add('loading');
    
    try {
        window.location.href = 'index.html';
    } catch (error) {
        setTimeout(() => {
            window.history.back();
            backBtn.classList.remove('loading');
        }, 300);
    }
}

// Toggle between login and signup forms
function showLogin() {
    const loginForm = document.querySelector('.login-form');
    const signupForm = document.querySelector('.signup-form');
    const schoolForm = document.querySelector('.school-form');
    const loginBtn = document.querySelector('.toggle-btn:first-child');
    const signupBtn = document.querySelector('.toggle-btn:last-child');
    
    loginForm.classList.add('active');
    signupForm.classList.remove('active');
    schoolForm.classList.remove('active');
    loginBtn.classList.add('active');
    signupBtn.classList.remove('active');
    
    loginForm.reset();
    hidePasswordStrength();
}

function showSignup() {
    const loginForm = document.querySelector('.login-form');
    const signupForm = document.querySelector('.signup-form');
    const schoolForm = document.querySelector('.school-form');
    const loginBtn = document.querySelector('.toggle-btn:first-child');
    const signupBtn = document.querySelector('.toggle-btn:last-child');
    
    signupForm.classList.add('active');
    loginForm.classList.remove('active');
    schoolForm.classList.remove('active');
    signupBtn.classList.add('active');
    loginBtn.classList.remove('active');
    
    signupForm.reset();
    hidePasswordStrength();
}

function showSchoolLogin() {
    const loginForm = document.querySelector('.login-form');
    const signupForm = document.querySelector('.signup-form');
    const schoolForm = document.querySelector('.school-form');
    const loginBtn = document.querySelector('.toggle-btn:first-child');
    const signupBtn = document.querySelector('.toggle-btn:last-child');
    
    schoolForm.classList.add('active');
    loginForm.classList.remove('active');
    signupForm.classList.remove('active');
    loginBtn.classList.remove('active');
    signupBtn.classList.remove('active');
    
    schoolForm.reset();
    hidePasswordStrength();
}

// Password strength checker
function checkPasswordStrength() {
    const password = document.getElementById('signupPassword').value;
    const strengthContainer = document.getElementById('passwordStrength');
    const strengthFill = document.getElementById('strengthFill');
    const strengthText = document.getElementById('strengthText');
    
    if (password.length === 0) {
        hidePasswordStrength();
        return;
    }
    
    strengthContainer.style.display = 'block';
    
    let score = 0;
    if (password.length >= 8) score += 1;
    if (password.length >= 12) score += 1;
    if (/[a-z]/.test(password)) score += 1;
    if (/[A-Z]/.test(password)) score += 1;
    if (/[0-9]/.test(password)) score += 1;
    if (/[^A-Za-z0-9]/.test(password)) score += 1;
    
    strengthFill.className = 'strength-fill';
    strengthText.className = '';
    
    if (score <= 2) {
        strengthFill.classList.add('weak');
        strengthText.classList.add('weak');
        strengthText.textContent = 'Weak password';
    } else if (score <= 4) {
        strengthFill.classList.add('medium');
        strengthText.classList.add('medium');
        strengthText.textContent = 'Medium strength';
    } else {
        strengthFill.classList.add('strong');
        strengthText.classList.add('strong');
        strengthText.textContent = 'Strong password';
    }
}

function hidePasswordStrength() {
    const strengthContainer = document.getElementById('passwordStrength');
    if (strengthContainer) {
        strengthContainer.style.display = 'none';
    }
}

// Form validation
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function validatePassword(password) {
    return password.length >= 8;
}

function showError(input, message) {
    const existingError = input.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    input.style.borderColor = '#ff4757';
    input.style.backgroundColor = '#ffe6e6';
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = 'color: #ff4757; font-size: 0.85rem; margin-top: 5px; font-weight: 500;';
    errorDiv.textContent = message;
    input.parentNode.appendChild(errorDiv);
    
    input.focus();
}

function clearError(input) {
    const errorMessage = input.parentNode.querySelector('.error-message');
    if (errorMessage) {
        errorMessage.remove();
    }
    
    input.style.borderColor = '#aaa';
    input.style.backgroundColor = '#c8c8c8';
}

// API helper function
async function apiCall(endpoint, data, method = 'POST') {
    try {
        // Support both '/auth/...' and already-prefixed '/api/...'
        const url = endpoint.startsWith('/api/')
            ? endpoint
            : `/api${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data),
            credentials: 'include'
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }
        
        return result;
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Social login functions
async function socialLogin(provider) {
    const button = event.target.closest('.social-btn');
    button.classList.add('loading');
    
    try {
        if (provider === 'google') {
            // Redirect to Google OAuth
            window.location.href = '/api/auth/google';
        } else {
            // Simulate other social login process
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // For demo purposes, create a mock social login
            const mockEmail = 'student@school.edu';
            const mockName = 'School Student';
            
            const result = await apiCall('/api/auth/login', {
                email: mockEmail,
                name: mockName,
                login_method: provider
            });
            
            showNotification('Social login successful!', 'success');
            
            setTimeout(() => {
                window.location.href = 'chatbot.html';
            }, 1000);
        }
        
    } catch (error) {
        button.classList.remove('loading');
        showNotification(`Social login failed: ${error.message}`, 'error');
    }
}

function showForgotPassword() {
    const email = prompt('Enter your email address for password reset:');
    if (email && validateEmail(email)) {
        // In a real app, you'd send this to your backend
        showNotification('Password reset link has been sent to your email!', 'info');
    } else if (email) {
        showNotification('Please enter a valid email address.', 'error');
    }
}

// Function to redirect to chatbot after successful authentication
function redirectToChatbot(userData) {
    const successMsg = document.createElement('div');
    successMsg.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #2ed573;
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    `;
    successMsg.textContent = `Welcome ${userData.name}! Redirecting...`;
    document.body.appendChild(successMsg);
    
    setTimeout(() => {
        window.location.href = 'chatbot.html';
    }, 1500);
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#ff4757' : type === 'success' ? '#2ed573' : '#888'};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        z-index: 10000;
        max-width: 300px;
        font-size: 14px;
        transition: all 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Handle Google OAuth callback
function handleGoogleCallback() {
    const urlParams = new URLSearchParams(window.location.search);
    const googleSuccess = urlParams.get('google_success');
    const googleError = urlParams.get('google_error');
    const reason = urlParams.get('reason');
    
    if (googleSuccess === 'true') {
        showNotification('Google login successful!', 'success');
        // Check if user is authenticated
        checkAuthStatus();
    } else if (googleError === 'true') {
        let errorMessage = 'Google login failed. Please try again.';
        if (reason === 'invalid_state') {
            errorMessage = 'Login session expired. Please try logging in again.';
        } else if (reason === 'expired') {
            errorMessage = 'Login request expired. Please try again.';
        }
        showNotification(errorMessage, 'error');
        // Clean up URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

// Check authentication status
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/auth/me', {
            method: 'GET',
            credentials: 'include'
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification(`Welcome back, ${result.user.name}!`, 'success');
            setTimeout(() => {
                window.location.href = 'chatbot.html';
            }, 1500);
        } else {
            // Clean up URL if not authenticated
            window.history.replaceState({}, document.title, window.location.pathname);
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        // Clean up URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

// Form submission handlers
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    const schoolForm = document.getElementById('schoolForm');
    
    // Handle Google OAuth callback on page load
    handleGoogleCallback();
    
    // Login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('loginEmail');
            const password = document.getElementById('loginPassword');
            const submitBtn = loginForm.querySelector('.submit-btn');
            
            clearError(email);
            clearError(password);
            
            let isValid = true;
            
            if (!email.value.trim()) {
                showError(email, 'Email is required');
                isValid = false;
            } else if (!validateEmail(email.value)) {
                showError(email, 'Please enter a valid email address');
                isValid = false;
            }
            
            if (!password.value.trim()) {
                showError(password, 'Password is required');
                isValid = false;
            }
            
            if (isValid) {
                submitBtn.classList.add('loading');
                submitBtn.textContent = 'SIGNING IN...';
                
                try {
                    const result = await apiCall('/api/auth/login', {
                        email: email.value.trim().toLowerCase(),
                        password: password.value,
                        login_method: 'form'
                    });
                    
                    showNotification('Login successful!', 'success');
                    redirectToChatbot(result.user);
                    
                } catch (error) {
                    submitBtn.classList.remove('loading');
                    submitBtn.textContent = 'LOGIN';
                    showError(password, error.message);
                }
            }
        });
    }
    
    // Signup form submission
    if (signupForm) {
        signupForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const name = document.getElementById('signupName');
            const email = document.getElementById('signupEmail');
            const password = document.getElementById('signupPassword');
            const confirmPassword = document.getElementById('confirmPassword');
            const submitBtn = signupForm.querySelector('.submit-btn');
            
            clearError(name);
            clearError(email);
            clearError(password);
            clearError(confirmPassword);
            
            let isValid = true;
            
            if (!name.value.trim()) {
                showError(name, 'Full name is required');
                isValid = false;
            } else if (name.value.trim().length < 2) {
                showError(name, 'Name must be at least 2 characters long');
                isValid = false;
            }
            
            if (!email.value.trim()) {
                showError(email, 'Email is required');
                isValid = false;
            } else if (!validateEmail(email.value)) {
                showError(email, 'Please enter a valid email address');
                isValid = false;
            }
            
            if (!password.value.trim()) {
                showError(password, 'Password is required');
                isValid = false;
            } else if (!validatePassword(password.value)) {
                showError(password, 'Password must be at least 8 characters long');
                isValid = false;
            }
            
            if (!confirmPassword.value.trim()) {
                showError(confirmPassword, 'Please confirm your password');
                isValid = false;
            } else if (password.value !== confirmPassword.value) {
                showError(confirmPassword, 'Passwords do not match');
                isValid = false;
            }
            
            if (isValid) {
                submitBtn.classList.add('loading');
                submitBtn.textContent = 'CREATING ACCOUNT...';
                
                try {
                    const result = await apiCall('/api/auth/signup', {
                        name: name.value.trim(),
                        email: email.value.trim().toLowerCase(),
                        password: password.value,
                        login_method: 'form'
                    });
                    
                    showNotification('Account created successfully!', 'success');
                    redirectToChatbot(result.user);
                    
                } catch (error) {
                    submitBtn.classList.remove('loading');
                    submitBtn.textContent = 'CREATE ACCOUNT';
                    
                    if (error.message.includes('already exists')) {
                        showError(email, error.message);
                    } else {
                        showNotification(`Signup failed: ${error.message}`, 'error');
                    }
                }
                         }
         });
     }
     
     // School form submission
     if (schoolForm) {
         schoolForm.addEventListener('submit', async function(e) {
             e.preventDefault();
             
             const name = document.getElementById('schoolName');
             const email = document.getElementById('schoolEmail');
             const couponCode = document.getElementById('schoolCoupon');
             const submitBtn = schoolForm.querySelector('.submit-btn');
             
             clearError(name);
             clearError(email);
             clearError(couponCode);
             
             let isValid = true;
             
             if (!name.value.trim()) {
                 showError(name, 'Full name is required');
                 isValid = false;
             } else if (name.value.trim().length < 2) {
                 showError(name, 'Name must be at least 2 characters long');
                 isValid = false;
             }
             
             if (!email.value.trim()) {
                 showError(email, 'School email is required');
                 isValid = false;
             } else if (!validateEmail(email.value)) {
                 showError(email, 'Please enter a valid email address');
                 isValid = false;
             }
             
             if (!couponCode.value.trim()) {
                 showError(couponCode, 'Coupon code is required');
                 isValid = false;
             }
             
             if (isValid) {
                 submitBtn.classList.add('loading');
                 submitBtn.textContent = 'LOGGING IN...';
                 
                 try {
                     const result = await apiCall('/api/auth/login', {
                         name: name.value.trim(),
                         email: email.value.trim().toLowerCase(),
                         coupon_code: couponCode.value.trim().toUpperCase(),
                         login_method: 'coupon'
                     });
                     
                     showNotification('School login successful!', 'success');
                     redirectToChatbot(result.user);
                     
                 } catch (error) {
                     submitBtn.classList.remove('loading');
                     submitBtn.textContent = 'LOGIN WITH SCHOOL ID';
                     
                     if (error.message.includes('Invalid coupon code')) {
                         showError(couponCode, 'Invalid coupon code');
                     } else if (error.message.includes('not valid for')) {
                         showError(email, 'Coupon code is not valid for this email domain');
                     } else if (error.message.includes('expired')) {
                         showError(couponCode, 'Coupon code has expired');
                     } else {
                         showNotification(`School login failed: ${error.message}`, 'error');
                     }
                 }
             }
         });
     }
     
     // Real-time validation
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.style.borderColor === 'rgb(255, 71, 87)') {
                clearError(this);
            }
        });
        
        input.addEventListener('blur', function() {
            if (this.value.trim() && this.type === 'email' && !validateEmail(this.value)) {
                showError(this, 'Please enter a valid email address');
            }
        });
    });
    
    // Password strength checking
    const signupPassword = document.getElementById('signupPassword');
    if (signupPassword) {
        signupPassword.addEventListener('input', checkPasswordStrength);
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            goToHomepage();
        }
        
        if (e.altKey) {
            if (e.key === 'l' || e.key === 'L') {
                showLogin();
                e.preventDefault();
            } else if (e.key === 's' || e.key === 'S') {
                showSignup();
                e.preventDefault();
            }
        }
    });
    
    // Add ripple effect to buttons
    function addRippleEffect(button) {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            ripple.classList.add('ripple');
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    }
    
    // Apply ripple effect to all buttons
    document.querySelectorAll('.submit-btn, .social-btn, .toggle-btn').forEach(addRippleEffect);
    
    // Add CSS for ripple animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
});