// Homepage Navigation
function goToAuth() {
    window.location.href = 'auth.html';
}

// Sliding Page (About)
function openSlidingPage() {
    document.getElementById('slidingPage').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeSlidingPage() {
    document.getElementById('slidingPage').classList.remove('active');
    document.body.style.overflow = 'auto';
}

// Ripple effect function
function addRippleEffect(button) {
    button.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.height, rect.width);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');
        
        // Make button position relative if not already
        const computedStyle = window.getComputedStyle(button);
        if (computedStyle.position === 'static') {
            button.style.position = 'relative';
        }
        
        button.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    });
}

// Close sliding page when clicking outside
document.addEventListener('DOMContentLoaded', function() {
    const slidingPage = document.getElementById('slidingPage');
    
    slidingPage.addEventListener('click', function(e) {
        if (e.target === this) {
            closeSlidingPage();
        }
    });
    
    // Handle sliding page transitions
    slidingPage.addEventListener('transitionend', function() {
        if (slidingPage.classList.contains('active')) {
            // Page is now fully visible, enable smooth scrolling
            slidingPage.style.scrollBehavior = 'smooth';
        } else {
           // Page is hidden, reset scroll position
            slidingPage.scrollTop = 0;
            slidingPage.style.scrollBehavior = 'auto';
        }
    });
    
    // Add ripple effects to buttons
    document.querySelectorAll('.start-btn').forEach(addRippleEffect);
});

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeSlidingPage();
    }
});