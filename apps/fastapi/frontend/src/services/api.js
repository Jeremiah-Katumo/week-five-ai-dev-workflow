import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8000" });

export const uploadCSV = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return API.post("/predict", formData);
};
