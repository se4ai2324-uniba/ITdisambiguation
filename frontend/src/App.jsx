import { useEffect, useState } from 'react'
import { useGetModels } from './services/ContentManager'
import OutputContainer from './OutputContainer';
import './App.css'
import ModelContainer from './ModelContainer';
import 'bootstrap'
import logo_it from './assets/logo_home.png';
import logo_uniba from './assets/logo_uniba.png';

export default function App() {
  
  const {models} = useGetModels();
  const [selectedModel, setSelectedModedl] = useState();

  useEffect(() => {
    if(models.length !== 0) setSelectedModedl(models[0])
  }, [models])

  return <div className="container-fluid mt-5">

    <div className="row my-3">
      <div className="col-12 text-center" >
      <h1>IT Disambiguation Team</h1>

        <div className="center-image">
          <img src={logo_it} />
          <img src={logo_uniba} />
        </div>
      </div>
    </div>

    <div className="row">
      
      <div className="col-6" id="div1">

        <div className="card shadow">
          <div className="card-body">
            <h5 className="card-title">Input</h5>
           

            {models.length === 0 ? 
              <div className="spinner-border" role="status">
                <span className="visually-hidden">Loading...</span>
              </div> :
              <div>
                {models.map((m,i) => 
                  <button className="btn btn-outline-secondary mx-2" key={i} type="button" onClick={() => setSelectedModedl(m)}>
                    {m}
                  </button>
                )}
              </div>
            }

            <ModelContainer modelName={selectedModel}/>
          </div>
        </div>
        


      </div>

      <div className="col-6">
        <div className="card shadow" id="div2">
          <div className="card-body">
            <h5 className="card-title">Output</h5>
            
            <OutputContainer/>
          </div>
        </div>
      </div>

    </div>
  </div>
}
