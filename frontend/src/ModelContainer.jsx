import { useState } from 'react'
import { useGetModelInfos } from './services/ContentManager'
import './ModelContainer.css'
import ACTIONS from './constants/actions';
import PredictImagesInput from './components/PredictImagesInput';
import PredictContextInput from './components/PredictContextInput';

export default function ModelContainer({modelName}) {

  const [selectedAction, setSelectedAction] = useState(ACTIONS.PREDICT_IMAGES);
  const {modelInfo} = useGetModelInfos(modelName);

  return <div>
    <div className="info-container">
      <div className="info-row">
        <strong>Description:</strong> {modelInfo?.description}
      </div>
      <div className="info-row">
        <strong>Typical usage:</strong> {modelInfo?.typical_usage}
      </div>
      <div className="info-row">
        <strong>Parameters number:</strong> {modelInfo?.n_parameters}
      </div>     
    </div>

    <div>
      <button key="predict_images" className='button' type="button" onClick={() => setSelectedAction(ACTIONS.PREDICT_IMAGES)}>
        {ACTIONS.PREDICT_IMAGES}
      </button>
      <button key="predict_context" className='button' type="button" onClick={() => setSelectedAction(ACTIONS.PREDICT_CONTEXT)}>
        {ACTIONS.PREDICT_CONTEXT}
      </button>
    </div>

    <div className='input_prediction_container'>
      {selectedAction === ACTIONS.PREDICT_IMAGES && <PredictImagesInput modelName={modelName}/>}
    {selectedAction === ACTIONS.PREDICT_CONTEXT && <PredictContextInput modelName={modelName}/>}
    </div>
  </div>
}