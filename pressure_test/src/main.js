// main.ts
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from '@/router'
import axios from "./api/apiClient.js";
import { createPinia } from 'pinia'
import Echarts from "vue-echarts"
import * as echarts from "echarts"
 
//导入持久化插件
import {createPersistedState} from'pinia-persistedstate-plugin'
//暗黑模式样式 1
import 'element-plus/theme-chalk/dark/css-vars.css'
import locale from 'element-plus/dist/locale/zh-cn.js'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'


const app = createApp(App)
const pinia = createPinia()
const persist = createPersistedState()
//pinia使用持久化插件
pinia.use(persist)
app.use(pinia)
app.use(router)
app.use(ElementPlus)
app.use(ElementPlus,{locale})
app.provide("$axios", axios);
app.mount('#app')
app.component("v-chart", Echarts)
 
app.config.globalProperties.$echarts = echarts
app.config.globalProperties.$axios = axios;

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }