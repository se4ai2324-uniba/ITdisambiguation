import React from "react";

export const PredictContext = React.createContext();

export const PredictContextProvider = ({ children }) => {
    const [output, setOutput] = React.useState({});

    return (
        <PredictContext.Provider value={{ output, setOutput }}>
            {children}
        </PredictContext.Provider>
    );
};