// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {

    // Get references to elements
    const form = document.getElementById('predictForm');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadPlaceholder = document.querySelector('.upload-placeholder'); // Get the placeholder div
    const loadingSpinner = document.querySelector('.loading-spinner');
    const errorMessageDiv = document.querySelector('.error-message'); // Use the div itself
    const predictionCard = document.querySelector('.prediction-card');
    const predictionResult = document.getElementById('predictionResult');
    const melanomaWarning = document.getElementById('melanomaWarning');
    const resultImage = document.getElementById('resultImage');
    const gradCamImage = document.getElementById('gradCamImage');
    const metadataSection = document.getElementById('processedMetadataSection');
    const metadataContent = document.getElementById('metadataContent');
    const resetButton = document.getElementById('resetButton'); // Get reset button

    // --- Image Preview --- 
    if (imageInput && imagePreview && uploadPlaceholder) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    uploadPlaceholder.style.display = 'none'; // Hide placeholder
                }
                reader.readAsDataURL(file);
            } else {
                imagePreview.style.display = 'none'; // Hide preview if no file
                uploadPlaceholder.style.display = 'block'; // Show placeholder
                imagePreview.src = ''; // Clear src
            }
        });
    }

    // --- Form Submission --- 
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Bootstrap validation
            if (!form.checkValidity()) {
                e.stopPropagation();
                form.classList.add('was-validated');
                return;
            }
            form.classList.remove('was-validated'); // Reset validation state if proceeding

            // Reset UI states
            if(loadingSpinner) loadingSpinner.style.display = 'block';
            if(errorMessageDiv) errorMessageDiv.style.display = 'none';
            if(predictionCard) predictionCard.style.display = 'none';
            if(melanomaWarning) melanomaWarning.style.display = 'none';
            errorMessageDiv.textContent = ''; // Clear previous errors

            try {
                const formData = new FormData(form);
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                // Check if response is OK (status in the range 200-299)
                if (!response.ok) {
                     let errorText = `HTTP error! Status: ${response.status}`;
                    try {
                        // Try to get more specific error from JSON response body
                        const errorJson = await response.json(); 
                        if (errorJson && errorJson.error) {
                            errorText = errorJson.error; 
                        }
                    } catch (jsonError) {
                        // If response is not JSON or error parsing it, use the status text
                        errorText = response.statusText || errorText; 
                    }
                    throw new Error(errorText);
                }

                const result = await response.json();
                
                // Check for application-specific error in the JSON
                if (result.error) {
                    throw new Error(result.error);
                }

                // --- Display Results --- 
                if (predictionResult && result.predictions) {
                    let resultHTML = '<div class="list-group list-group-flush mb-3">'; // Use flush for cleaner look inside card
                    result.predictions.forEach((pred, index) => {
                        const probability = parseFloat(pred.probability) * 100;
                        const badgeClass = pred.is_melanoma ? 'bg-danger' : 'bg-success'; // Use danger/success
                        
                        resultHTML += `
                            <div class="list-group-item d-flex justify-content-between align-items-center">
                                <span>${pred.class_name}</span>
                                <span class="badge ${badgeClass} rounded-pill">${probability.toFixed(1)}%</span>
                            </div>
                        `;
                    });
                    resultHTML += '</div>';
                    predictionResult.innerHTML = resultHTML;
                }
                
                // Display original image in results
                if (resultImage && imagePreview.src) {
                    resultImage.src = imagePreview.src;
                    resultImage.style.display = 'block';
                }

                // Display Grad-CAM if available
                if (gradCamImage && result.grad_cam_image) {
                    gradCamImage.src = result.grad_cam_image; // Already base64
                    gradCamImage.style.display = 'block';
                } else if (gradCamImage) {
                    gradCamImage.style.display = 'none';
                }

                // Add melanoma-specific warning if probability is high
                if (melanomaWarning && result.melanoma_probability) {
                    const melanomaProb = parseFloat(result.melanoma_probability) * 100;
                    // Adjust threshold as needed
                    if (melanomaProb > 20) { 
                        melanomaWarning.style.display = 'block';
                    } else {
                         melanomaWarning.style.display = 'none';
                    }
                }

                // Display processed metadata if available
                if (metadataSection && metadataContent && result.processed_metadata && result.original_metadata) {
                    let metaHTML = '<ul class="list-unstyled small">';
                    if (result.original_metadata.age !== undefined) metaHTML += `<li><strong>Original Age:</strong> ${result.original_metadata.age}</li>`;
                    if (result.processed_metadata.age_normalized !== undefined) metaHTML += `<li><strong>Normalized Age:</strong> ${result.processed_metadata.age_normalized.toFixed(4)}</li>`;
                    if (result.original_metadata.sex !== undefined) metaHTML += `<li><strong>Original Sex:</strong> ${result.original_metadata.sex}</li>`;
                    if (result.processed_metadata.sex_processed !== undefined) metaHTML += `<li><strong>Processed Sex (0=F, 1=M):</strong> ${result.processed_metadata.sex_processed}</li>`;
                    if (result.original_metadata.anatom_site_general !== undefined) metaHTML += `<li><strong>Original Location:</strong> ${result.original_metadata.anatom_site_general || 'Not specified'}</li>`;
                    if (result.processed_metadata.location_one_hot !== undefined && result.processed_metadata.location_columns !== undefined) {
                        metaHTML += `<li><strong>Location Vector (One-Hot):</strong><br><small>[${result.processed_metadata.location_columns.join(', ')}]</small><br><code>[${result.processed_metadata.location_one_hot.join(', ')}]</code></li>`;
                    }
                    if (result.processed_metadata.n_images_log !== undefined) metaHTML += `<li><strong>Log(n_images+1):</strong> ${result.processed_metadata.n_images_log.toFixed(4)}</li>`;
                    if (result.processed_metadata.image_size_log !== undefined) metaHTML += `<li><strong>Log(image_size):</strong> ${result.processed_metadata.image_size_log.toFixed(4)}</li>`;
                    metaHTML += '</ul>';
                    metadataContent.innerHTML = metaHTML;
                    metadataSection.style.display = 'block';
                } else if (metadataSection) {
                    metadataSection.style.display = 'none';
                    if(metadataContent) metadataContent.innerHTML = ''; // Clear previous content
                }

                if(predictionCard) predictionCard.style.display = 'block'; // Show results card

            } catch (error) {
                console.error('Error during prediction:', error);
                if(errorMessageDiv) { 
                    errorMessageDiv.textContent = error.message || 'An unexpected error occurred. Please try again.';
                    errorMessageDiv.style.display = 'block';
                } 
            } finally {
                if(loadingSpinner) loadingSpinner.style.display = 'none'; // Hide spinner regardless of outcome
            }
        });
    }

    // --- Reset Button Logic --- 
    if (resetButton && form) {
        resetButton.addEventListener('click', function() {
            // Reset the form fields to default values
            form.reset(); 

            // Manually hide/show elements and clear states
            if (imagePreview) {
                imagePreview.style.display = 'none';
                imagePreview.src = ''; // Clear src
            }
            if (uploadPlaceholder) {
                uploadPlaceholder.style.display = 'block'; // Show placeholder
            }
            if (predictionCard) {
                predictionCard.style.display = 'none';
            }
            if (errorMessageDiv) {
                errorMessageDiv.style.display = 'none';
                errorMessageDiv.textContent = '';
            }
             if (melanomaWarning) {
                melanomaWarning.style.display = 'none';
            }
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            if (metadataSection) {
                metadataSection.style.display = 'none';
            }
            if (metadataContent) {
                metadataContent.innerHTML = '';
            }

            // Remove Bootstrap validation classes
            form.classList.remove('was-validated');
            form.querySelectorAll('.is-valid, .is-invalid').forEach(el => {
                el.classList.remove('is-valid', 'is-invalid');
            });
        });
    }

    // --- Reset validation on input --- 
    if (form) {
        form.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('input', () => {
                // Remove validation state only if it was previously triggered
                if (form.classList.contains('was-validated')) {
                   if (input.checkValidity()) {
                        input.classList.remove('is-invalid');
                        input.classList.add('is-valid'); // Optional: show valid state
                    } else {
                        input.classList.remove('is-valid');
                        input.classList.add('is-invalid');
                    }
                    // Check overall form validity to potentially remove the form-level class
                    // Note: This doesn't re-trigger validation on all fields
                    // let isFormStillInvalid = Array.from(form.elements).some(el => !el.checkValidity());
                    // if (!isFormStillInvalid) { 
                    //     form.classList.remove('was-validated');
                    // } 
                }
            });
        });
    }

}); // End of DOMContentLoaded 