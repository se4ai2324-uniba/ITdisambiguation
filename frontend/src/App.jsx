import { useEffect, useState } from 'react'
import { useGetModels } from './services/ContentManager'
import './App.css'
import ModelContainer from './ModelContainer';

export default function App() {
  
  const {models} = useGetModels();
  const [selectedModel, setSelectedModedl] = useState();

  useEffect(() => {
    if(models.length !== 0) setSelectedModedl(models[0])
  }, [models])

  return <div className="page-container">
    <div className="row">
      
      <div className="column" id="div1">
        
        {models.length === 0 ? 
          "Loading..." :
          <div>
            {models.map((m,i) => 
              <button key={i} type="button" onClick={() => setSelectedModedl(m)}>
                {m}
              </button>
            )}
          </div>
        }

        <ModelContainer modelName={selectedModel}/>

      </div>

      <div className="column" id="div2">
        Output content
      </div>
      
    </div>
    
  </div>
}
