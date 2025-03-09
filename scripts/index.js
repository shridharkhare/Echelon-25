import { setInitialTheme } from "./utils/theme.js";

setInitialTheme();

// Event listeners for landing page
const lobbyButton = document.getElementById("lobby-btn");

lobbyButton.addEventListener("click", () => {
    window.location.href = "./lobby.html";
});