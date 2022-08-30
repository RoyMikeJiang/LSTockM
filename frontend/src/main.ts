import { createApp, reactive } from "vue";
import App from "./App.vue";

// import "~/styles/element/index.scss";

import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

// or use cdn, uncomment cdn link in `index.html`

import "~/styles/index.scss";
import 'uno.css'

import router from './router'

import { useCookies } from "vue3-cookies";
const { cookies } = useCookies();
import axios from 'axios'

// //引入echarts
// import * as echarts from  'echarts'
// //引入vue-echarts组件
// import VueECharts from 'vue-echarts'

const app = createApp(App);
app.use(ElementPlus);
app.use(router);
// app.component('v-chart', VueECharts)

// const BasicUrl = 'http://localhost:3000/api'
// const AccountInfo = {
//     login : false,
//     account_id : 0,
//     username : ''
// }

// app.config.globalProperties.BasicUrl = 'http://localhost:3000/api'
app.config.globalProperties.BasicUrl = 'https://lstockm.roymikejiang.tech/api'
app.config.globalProperties.AccountInfo = reactive({
    login : false,
    account_id : 0,
    username : ''
})

app.mount("#app");
