import {baseUrl} from "../constants/network";
import axios from 'redaxios';

export const MODELS = "models"
export const PREDICT_CONTEXT = "predict_context"
export const PREDICT_IMAGES = "predict_images"

export function get(url, config = {elem: "", params: {}, header: {}}) {
	return axios.get(getUrl(url, config.elem), getConfig(config)).then(({data}) => data.data);
}

export function post(url, config = {elem: "", body: {}, params: {}, header: {}}) {
	return axios.post(getUrl(url, config.elem), config.body, getConfig(config));
}

export function put(url, config = {elem: "", body: {}, params: {}, header: {}}) {
	return axios.put(getUrl(url, config.elem), config.body, getConfig(config));
}

export function deleteElem(url, config = {elem: "", body: {}, params: {}, header: {}}) {
	return axios.delete(getUrl(url, config.elem), getConfig(config));
}

export function download(url) {
	return axios.get(getUrl(url), {responseType: 'blob', ...getConfig({})});
}

export function getErrorMessage(e) {
	let error = "Unknown error";
	if (e.response) error = e.response.data.error;
	else if (e.data) error = e.data.message || e.data.error;
	else error = e.message || e.error;

	return error;
}

function getConfig({params = {}, headers = {}}) {
	return {
		params,
		headers: {
            ...headers
        }
	};
}

function getUrl(url, elem) {
	return elem ? `${baseUrl}${url}/${elem}` : `${baseUrl}${url}`;
}

