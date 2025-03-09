// Theme switcher
const darkModeButton = document.getElementById('theme-button');
const darkModeIcon = darkModeButton.firstElementChild;

// Set the starting theme
export function setInitialTheme() {
    const prefferedTheme = window?.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';

    // Update html attribute (data-theme)
    document.documentElement.setAttribute('data-theme', prefferedTheme === 'dark' ? 'dark' : 'light');
    // Update icon
    if (prefferedTheme === 'dark') {
        darkModeIcon.classList.add('fa-sun');
    } else {
        darkModeIcon.classList.add('fa-moon');
    }
}

// Set the button text according to the current theme
darkModeButton.addEventListener('click', () => {
    const theme = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'light' : 'dark');
    darkModeIcon.classList.toggle('fa-moon');
    darkModeIcon.classList.toggle('fa-sun');
});
