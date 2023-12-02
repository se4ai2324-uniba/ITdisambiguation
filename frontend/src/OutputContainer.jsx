import React from "react";
import { PredictContext } from "./services/PredictionContextProvider";
import ACTIONS from "./constants/actions";

export default function OutputContainer() {

    const { output } = React.useContext(PredictContext);

    if(output === "Loading...") 
        return  <div className="spinner-border" role="status">
                    <span className="visually-hidden">Loading...</span>
                </div>
   
    if(output.type === ACTIONS.PREDICT_IMAGES) 
        return <div className="info-container">
            <div className="info-row">
                <h2><strong>{output.type}</strong></h2>
            </div>
            <div className="info-row">
                Model Name: <strong>{output.model_name}</strong>
            </div>
            <div className="info-row">
                Target Word: <strong>{output.target_word}</strong>
            </div>     
            <div className="info-row">
                Context: <strong>{output.context}</strong>
            </div>     
            <div className="info-row">
                Prediction score: <strong>{output.predicted_score.toFixed(3)}</strong>
            </div>  
            <p>
                <div>
                    {output.images.map((image, index) => 
                        <img src={URL.createObjectURL(image)} 
                            style={{ 
                                maxWidth: '100%', maxHeight: '200px', margin: "10px",
                                border: index === output.predicted_image_index ?
                                    "10px solid green" : "5px solid red" }}
                        />
                    )}
                </div>
            </p>   
        </div>

    if(output.type === ACTIONS.PREDICT_CONTEXT) 
        return <div className="info-container">
            <div className="info-row">
                <h2><strong>{output.type}</strong></h2>
            </div>
            <div className="info-row">
                Model Name: <strong>{output.model_name}</strong>
            </div>
            <div className="info-row">
                Target Word: <strong>{output.target_word}</strong>
            </div>     
            <div className="info-row">
                Contexts: <strong>{output.contexts}</strong>
            </div>     
            <div>
                <p>Selected Image:</p>
                <img
                    id="OutputImage" src={URL.createObjectURL(output.image)}  alt="OutputImage"
                    style={{ maxWidth: '100%', maxHeight: '200px' }}
                />
            </div> 
            <div className="info-row">
                Prediction score: <strong>{output.predicted_score.toFixed(3)}</strong>
            </div>  
            <div className="info-row">
                Predicted Context: <strong style={{color: "green"}}>
                 {output.predicted_context.charAt(0).toUpperCase() + output.predicted_context.slice(1)}
                </strong>
            </div>  
        </div>
    
    if(output.type === "error")
        return <div className="info-container">
            <div className="info-row">
                <h2 style={{color: "red"}}><strong>Error {output.status}</strong></h2>
            </div>
            {output?.data?.detail?.map(e => <div className="info-row">
                {e.msg} <strong>{e.loc[1]}</strong>
            </div>)}
        </div>

}