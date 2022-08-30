import {createRouter, createWebHashHistory} from "vue-router"
import {defineAsyncComponent} from 'vue'

// Register pages in the routes
const market = defineAsyncComponent(() => import("../components/Market.vue"))
const myStocks = defineAsyncComponent(() => import("../components/MyStocks.vue"))
const account = defineAsyncComponent(() => import("../components/Account.vue"))
const stockDetail = defineAsyncComponent(() => import("../components/StockDetail.vue"))
const predictRequest = defineAsyncComponent(() => import("../components/PredictRequest.vue"))
const login = defineAsyncComponent(() => import("../components/Login.vue"))

const routes = [
    {
        path: '/',
        redirect: '/market'
    },
    {
        path: '/market',
        name: 'Market',
        component: market
    },
    {
        path: '/mystocks',
        name: 'MyStocks',
        component: myStocks
    },
    {
        path: '/account',
        name: 'Account',
        component: account
    },
    {
        path: '/detail',
        name: 'StockDetail',
        component: stockDetail
    },
    {
        path: '/request',
        name: 'PredictRequest',
        component: predictRequest
    },
    {
        path: '/login',
        name: 'Login',
        component: login
    }
]

// Create router and expose it.
const router = createRouter({
    history: createWebHashHistory(),
    routes: routes
})
export default router;