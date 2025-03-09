document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const uploadButton = document.getElementById("uploadButton");

    imageInput.addEventListener("change", previewImage);
    uploadButton.addEventListener("click", uploadImage);
});

let uploadedImage; 

function previewImage(event) {
    let reader = new FileReader();
    reader.onload = function () {
        let output = document.getElementById("preview");
        output.src = reader.result;
        output.style.display = "block"; 
    };
    uploadedImage = event.target.files[0]; 
    reader.readAsDataURL(uploadedImage); 

}


async function uploadImage(event) {
    if (!uploadedImage) {
        alert("Please select an image to upload.");
        return;
    }

    let formData = new FormData();
    formData.append("file", uploadedImage);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (data.image_url) {
            document.getElementById("outputImage").src = data.image_url + "?t=" + new Date().getTime();
            document.getElementById("outputImage").style.display = "block";
        } else {
            alert("Error processing image");
        }
    } catch (error) {
        console.error("Error:", error);
    }
}