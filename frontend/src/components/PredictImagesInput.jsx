import React from 'react'
import { useState } from 'react'
import { post, MODELS, PREDICT_IMAGES } from '../services/Client';
import { PredictContext } from '../services/PredictionContextProvider';
import ACTIONS from '../constants/actions';

export default function PredictImagesInput({modelName}){

    const { setOutput } = React.useContext(PredictContext);
    const [targetWord, setTargetWord] = useState('');
    const [context, setContext] = useState('');
    const [imageFiles, setImageFiles] = useState(null);

    const predictImages = () => {
        let bodyFormData = new FormData();
        bodyFormData.append('target_word', targetWord);
        bodyFormData.append('context', context);
        imageFiles?.map((file, index) => bodyFormData.append('images', file))

        setOutput("Loading...")
        post(`${MODELS}/${modelName}/${PREDICT_IMAGES}`, {body: bodyFormData})
            .then(({data}) => setOutput({
                type: ACTIONS.PREDICT_IMAGES,
                images: imageFiles,
                ...data.data
            }))
            .catch((error) => setOutput({
                ...error,
                type: "error",
            }))
    }

    // Event handler for file input change
    const handleImageChange = (event) => {
        const selectedFiles = Array.from(event.target.files);

        if (selectedFiles.length > 0) {
            // Set the selected files to state
            setImageFiles(selectedFiles);

            // Optionally, you can display the image previews
            selectedFiles.forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    // Display the image preview using the FileReader result
                    const previewImage = document.getElementById(`previewImage${index}`);
                    previewImage.src = e.target.result;
                };
                reader.readAsDataURL(file);
            });
        }
    };
    
    return <div>
        <p>
            <div class="input-group mb-3">
                <span class="input-group-text" id="target-addon">Target Word</span>
                <input 
                    type="text" 
                    class="form-control" 
                    placeholder="Eagle" 
                    aria-describedby="target-addon"
                    id="target-word"
                    value={targetWord}
                    onChange={(event) => setTargetWord(event.target.value)}
                    />
            </div>

        </p>
        <p>
            <div class="input-group mb-3">
            <span class="input-group-text" id="context-addon">Context</span>
                <input
                    type="text" 
                    class="form-control" 
                    placeholder="Bird" 
                    aria-describedby="context-addon"
                    id="contexts"
                    value={context}
                    onChange={(event) => setContext(event.target.value)}
                /> 
            </div>
        </p>

        <p>
            <div>
                <div class="input-group mb-3">
                    <label class="input-group-text" for="imageInput">Target Images</label>
                    <input
                        type="file"
                        class="form-control"
                        id="imageInput"
                        accept="image/*"
                        onChange={handleImageChange}
                        multiple
                    />
                </div>

                <div key="images">
                    <p>Selected Images:</p>
                    {imageFiles && imageFiles.map((file, index) => (
                        <img
                            id={`previewImage${index}`}
                            src=""
                            alt="Preview"
                            style={{ maxWidth: '100%', maxHeight: '200px' }}
                        />
                    ))}
                </div>
            </div>
        </p>

        <div class="d-grid gap-2">
            <button class="btn btn-outline-success" key="send-predict-context" type="button" onClick={predictImages}>Send</button>
        </div>
    </div>
}