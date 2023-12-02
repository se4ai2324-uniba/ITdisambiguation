import { useGetModelInfos } from './services/ContentManager'
import './ModelContainer.css'
import ACTIONS from './constants/actions';
import PredictImagesInput from './components/PredictImagesInput';
import PredictContextInput from './components/PredictContextInput';


export default function ModelContainer({modelName}) {

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
        <strong>Parameters number:</strong> {modelInfo?.n_parameters?.toLocaleString('it-IT')}
      </div>     
    </div>


    <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist">
      <li class="nav-item" role="presentation">
        <button 
        class="nav-link active" 
        id="pills-predict-images-tab" 
        data-bs-toggle="pill" 
        data-bs-target="#pills-predict-images" 
        type="button" 
        role="tab" 
        aria-controls="pills-predict-images" 
        aria-selected="true">
          
          {ACTIONS.PREDICT_IMAGES}
        
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button 
        class="nav-link" 
        id="pills-predict-context-tab" 
        data-bs-toggle="pill" 
        data-bs-target="#pills-predict-context" 
        type="button" 
        role="tab" 
        aria-controls="pills-predict-context" 
        aria-selected="false">
          {ACTIONS.PREDICT_CONTEXT}
          </button>
      </li>
    </ul>
    <div class="tab-content" id="pills-tabContent">
      <div class="tab-pane fade show active" id="pills-predict-images" role="tabpanel" aria-labelledby="pills-predict-images-tab" tabindex="0"><PredictImagesInput modelName={modelName}/></div>
      <div class="tab-pane fade" id="pills-predict-context" role="tabpanel" aria-labelledby="pills-predict-context-tab" tabindex="0"><PredictContextInput modelName={modelName}/></div>
    </div>

  </div>
}