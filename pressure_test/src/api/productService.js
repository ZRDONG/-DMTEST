// src/api/productService.js

import apiClient from './apiClient';

export default {
  getParameter() {
    // 向 '/query_all' 路由发送 GET 请求
    return apiClient.get('/query_all');
  },
  gettpmC(){
    return apiClient.get('/get_image');
  }
};
