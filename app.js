document.addEventListener('DOMContentLoaded', () => {
    // Scroll reveal animation logic using Intersection Observer API
    const revealElements = document.querySelectorAll('.reveal');

    const revealOptions = {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    };

    const revealOnScroll = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                return;
            } else {
                entry.target.classList.add('active');
                observer.unobserve(entry.target);
            }
        });
    }, revealOptions);

    revealElements.forEach(el => {
        revealOnScroll.observe(el);
    });
    
    // Add staggered delay for the initial header cards load
    const headerCards = document.querySelectorAll('.header-card');
    headerCards.forEach((card, index) => {
        card.style.transitionDelay = `${0.3 + (index * 0.15)}s`;
    });
});
