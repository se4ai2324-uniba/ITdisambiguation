import { useState } from 'react'
import { post, MODELS, PREDICT_CONTEXT } from '../services/Client';

export default function PredictContextInput({modelName}){

    const [targetWord, setTargetWord] = useState('');
    const [contexts, setContexts] = useState('');
    const [imageFile, setImageFile] = useState(null);

    const predictContext = () => {

        let bodyFormData = new FormData();
        bodyFormData.append('target_word', targetWord);
        bodyFormData.append('contexts', contexts);
        bodyFormData.append('image', imageFile);

        post(`${MODELS}/${modelName}/${PREDICT_CONTEXT}`, {body: bodyFormData})
            .then(r => console.log(r))
    }

    console.log(imageFile)

    // Event handler for file input change
    const handleImageChange = (event) => {
        const selectedFile = event.target.files[0];

        if (selectedFile) {
            // Set the selected file to state
            setImageFile(selectedFile);

            // Optionally, you can display the image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                // Display the image preview using the FileReader result
                const previewImage = document.getElementById('previewImage');
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(selectedFile);
        }
    };

    return <div>
        <p>
            <label htmlFor="simpleInput">Target Word: </label>
            <input
                type="text"
                id="target-word"
                value={targetWord}
                onChange={(event) => setTargetWord(event.target.value)}
            />
        </p>
        <p>
            <label htmlFor="simpleInput">Contexts divided by commas: </label>
            <input
                type="text"
                id="contexts"
                value={contexts}
                onChange={(event) => setContexts(event.target.value)}
            /> 
        </p>

        <p>
            <label htmlFor="imageInput">Select Image: </label>
            <input
                type="file"
                id="imageInput"
                accept="image/*"
                onChange={handleImageChange}
            />
            {imageFile && (
                <div>
                <p>Selected Image:</p>
                <img
                    id="previewImage"
                    src=""
                    alt="Preview"
                    style={{ maxWidth: '100%', maxHeight: '200px' }}
                />
                </div>
            )}
        </p>

        <p>
            <button key="send-predict-context" type="button" onClick={predictContext}>Send</button>
        </p>
    </div>
}