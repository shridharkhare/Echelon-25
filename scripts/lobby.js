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


function uploadImage(event) {
    let formData = new FormData();
    formData.append("image", uploadedImage);

    fetch("http://127.0.0.1:5000/predict", { 
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("prediction").innerText = data.prediction;
        
        if (data.image) {
            let outputImg = document.getElementById("outputImage");
            outputImg.src = "data:image/png;base64," + data.image;
            outputImg.style.display = "block";
        }
    })
    .catch(error => console.error("Error:", error));
}
