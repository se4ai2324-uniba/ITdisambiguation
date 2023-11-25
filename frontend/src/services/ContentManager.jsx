import {useQuery} from "react-query";
import {get,MODELS} from "./Client";


export function useGetModels() {
	const {status, data, error} = useQuery([MODELS], () => get(MODELS), {staleTime: 300000});
	return {status, models: data ? data.model_names : [], error};
}

export function useGetModelInfos(modelName) {
	const {status, data, error} = useQuery([MODELS, modelName], () => get(MODELS, {elem: modelName}), {staleTime: Infinity});
	return {status, modelInfo: data || {}, error};
}