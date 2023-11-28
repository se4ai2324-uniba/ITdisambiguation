import React from 'react'
import { useState } from 'react'
import { post, MODELS, PREDICT_CONTEXT } from '../services/Client';
import { PredictContext } from '../services/PredictionContextProvider';
import ACTIONS from '../constants/actions';

export default function PredictContextInput({modelName}){

    const { setOutput } = React.useContext(PredictContext);
    const [targetWord, setTargetWord] = useState('');
    const [contexts, setContexts] = useState('');
    const [imageFile, setImageFile] = useState(null);

    const predictContext = () => {

        let bodyFormData = new FormData();
        bodyFormData.append('target_word', targetWord);
        bodyFormData.append('contexts', contexts);
        bodyFormData.append('image', imageFile);

        setOutput("Loading...")
        post(`${MODELS}/${modelName}/${PREDICT_CONTEXT}`, {body: bodyFormData})
            .then(({data}) => setOutput({
                type: ACTIONS.PREDICT_CONTEXT,
                image: imageFile,
                ...data.data
            }))
            .catch((error) => setOutput({
                ...error,
                type: "error"
            }))
    }

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
            <div class="input-group mb-3">
                <span class="input-group-text" id="target-addon">Target Word</span>
                <input 
                    type="text" 
                    class="form-control" 
                    placeholder="Aquila" 
                    aria-describedby="target-addon"
                    id="target-word"
                    value={targetWord}
                    onChange={(event) => setTargetWord(event.target.value)}
                    />
            </div>

        </p>
        <p>
            <div class="input-group mb-3">
            <span class="input-group-text" id="context-addon">Contexts (divided by commas) </span>
                <input
                    type="text" 
                    class="form-control" 
                    placeholder="bird, constellation stars" 
                    aria-describedby="context-addon"
                    id="contexts"
                    value={contexts}
                    onChange={(event) => setContexts(event.target.value)}
                /> 
            </div>
        </p>

        <p>
            <div class="input-group mb-3">
                <label class="input-group-text" for="imageInput">Target Image</label>
                <input
                type="file"
                class="form-control"
                id="imageInput"
                accept="image/*"
                onChange={handleImageChange}
                />
            </div>

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

        <div class="d-grid gap-2">
            <button class="btn btn-outline-success" key="send-predict-context" type="button" onClick={predictContext}>Send</button>
        </div>
    </div>
}