import { createRouter, createWebHashHistory} from 'vue-router'
 
const routes = [
  { path: '/', component: () => import('@/pressure/result.vue') ,redirect:'/data',children:[
  { path: '/data', component: () => import('@/pressure/data.vue') },
  { path: '/target', component: () => import('@/pressure/target.vue') },
  { path: '/parameter', component: () => import('@/pressure/parameter.vue') },
  { path: '/log', component: () => import('@/pressure/log.vue') }]}
]
 


const router = createRouter({
  history: createWebHashHistory(),
  routes
})
 
export default router