<template>
<div class="centerBox">
  <el-row>
    <el-col :span="24">
        <div class="primaryStockBasicInfo">
            {{stock_basic_info.stock_symbol}} - {{stock_basic_info.company_name}}
            &nbsp;&nbsp;&nbsp;&nbsp;
            <el-button :icon="Star" :type="watchlist_button_type" @click="handleWatchlist()" :disabled="watchlist_button_loading" round>{{watchlist_button_text}}</el-button>
        </div>
    </el-col>
  </el-row>
  <el-row>
    <el-col :span="24">
        <div class="secondaryStockBasicInfo">
            {{stock_basic_info.exchange_name}}&nbsp;&nbsp;&nbsp;&nbsp;Currency in {{stock_basic_info.currency}}
        </div>
    </el-col>
  </el-row>
  <el-row>
    <el-col :span="24">
        <div class="priceInfo">
            {{stock_basic_info.current_price.toFixed(4)}}
            <text :class="price_change >= 0 ? 'green-price-change' : 'red-price-change'">{{(price_change<0?"":"+") + price_change.toFixed(4)}}</text>
        </div>
        <div class="secondaryStockBasicInfo">
            At {{stock_basic_info.current_price_time}}
        </div>
    </el-col>
  </el-row>
  <el-row :gutter="20"> 
  <el-col :span="2" class="multiButtons">
        <el-row>
            <el-button class="flex justify-space-between mb-4 flex-wrap gap-4" size="large" text bg style="width:80px" @click="getDiagramData(7)">7 Days</el-button>
        </el-row>
        <el-row>
            <el-button class="flex justify-space-between mb-4 flex-wrap gap-4" size="large" text bg style="width:80px" @click="getDiagramData(14)">14 Days</el-button>
        </el-row>
        <el-row>
            <el-button class="flex justify-space-between mb-4 flex-wrap gap-4" size="large" text bg style="width:80px" @click="getDiagramData(30)">30 Days</el-button>
        </el-row>
        <el-row>
            <el-button class="flex justify-space-between mb-4 flex-wrap gap-4" size="large" text bg style="width:80px" @click="getDiagramData(90)">90 Days</el-button>
        </el-row>
        <el-row>
            <el-button class="flex justify-space-between mb-4 flex-wrap gap-4" size="large" text bg style="width:80px" @click="getDiagramData(356)">1 Year</el-button>
        </el-row>
        <el-row>
            <el-switch v-model="predict_show" active-text="Predict" size="default"/>
        </el-row>
    </el-col>   
    <el-col :span="14" v-loading="loading">
        <v-chart class="diagram" :option="diagram_config"></v-chart>
    </el-col>
    
      <el-col :span="4" class="statistics">
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Ebitda&nbsp;Margins: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.ebitdaMargins}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Return&nbsp;On&nbsp;Assets: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.returnOnAssets}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>PEG Ratio:  </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.pegRatio}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Value&nbsp;At&nbsp;Risk:  </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.valueAtRisk}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Price&nbsp;To&nbsp;Book:  </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.priceToBook}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Volume: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.volume}}</div></el-col>
        </el-row>
    </el-col>
    <el-col :span="4" class="statistics">
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Operating&nbsp;Margins: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.operatingMargins}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Return&nbsp;On&nbsp;Equity: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.returnOnEquity}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Quick Ratio:  </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.quickRatio}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Beta:  </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.beta}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Bid: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.bid}} x {{stock_price_statistics.bidSize}}</div></el-col>
        </el-row>
        <el-row>
            <el-col :span="12" class="statistics_left"><div>Ask: </div></el-col>
            <el-col :span="12" class="statistics_right"><div>{{stock_price_statistics.ask}} x {{stock_price_statistics.askSize}}</div></el-col>
        </el-row>
    </el-col>
  </el-row>
  <el-row :gutter="20">
    <el-col :span="2"></el-col>
    <el-col :span="12">
    <div class="block text-center" m="t-4">
    <el-carousel trigger="click" height="120px" :interval="10000">
      <el-carousel-item v-for="item, index in suggestions" :key="index" v-if="suggestions.length">
        <div>
            <h3 class="sug-text">{{ item.label }}</h3>
            <div class="sug-text">{{ item.suggestion }}</div>
        </div>
      </el-carousel-item>
    </el-carousel>
    </div>
    </el-col>
    <el-col :span="2"></el-col>
    <el-col :span="6">
    <div class="block text-center" m="t-4">
    <el-carousel trigger="click" height="120px" :interval="10000" v-if="remote_suggestions.length">
      <el-carousel-item v-for="item, index in remote_suggestions" :key="index">
        <div>
            <h3 class="sug-text">{{ item.label }}</h3>
            <div class="sug-text">{{ item.suggestion }}</div>
        </div>
      </el-carousel-item>
    </el-carousel>
    </div>
    </el-col>
    <el-col :span="2"></el-col>
    </el-row>
</div>
</template>

<script lang="ts">
import { color, List, use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import {Star} from '@element-plus/icons-vue'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components';
import VChart, { THEME_KEY } from 'vue-echarts';
use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
]);
import { useRoute, useRouter } from 'vue-router'
import { ref, reactive, onMounted, getCurrentInstance , computed, computed} from 'vue'
import axios from 'axios'
export default {
    name: 'StockDetail',
    components: {VChart},
    setup(){
        const router = useRouter()
        const route = useRoute()
        const { proxy } = getCurrentInstance()
        const stock_basic_info = reactive({
            stock_id : route.query['stock_id'],
            stock_symbol : route.query['stock_symbol'],
            current_price : 0,
            current_price_time : "Loading..." 
        })
        const stock_price_statistics = ref({})
        const diagram_data = ref([])
        const diagram_date = ref([])
        const loading = ref(false)
        const predict_data = reactive({
            price : [],
            date : []
        })
        const predict_show = ref(false)
        const inwatchlist = ref(false)
        const watchlist_button_loading = ref(false)
        const watchlist_button_type = computed(()=>{
            if(inwatchlist.value){
                return "primary"
            }else{
                return "default"
            }
        })
        const watchlist_button_text = computed(()=>{
            if(inwatchlist.value){
                return "Remove from Watchlist"
            }else{
                return "Add to Watchlist"
            }
        })
        const remote_suggestions = reactive([])
        const suggestions = computed(()=>{
            var data_suggestions = []
            var labels = []
            var temp_suggestions = []
            
            labels.unshift('Analysis on Beta')
            if(stock_price_statistics.value.beta>=0.7){
                temp_suggestions.unshift('This stock has a relatively high beta, which tends to move with more momentum than others. You may have larger possibility to get more profit, while you have to tolerate more risk.')
            }else if(stock_price_statistics.value.beta<=0.3){
                temp_suggestions.unshift('This stock has a relatively low beta, which tends to move with less momentum than others. You may have a stable profit, while the possibility to get big profit is small.')
            }else{
                temp_suggestions.unshift('This stock has a moderate beta.')
            }
            labels.unshift('Analysis on PEG Ratio')
            if(stock_price_statistics.value.pegRatio>=0.7){
                temp_suggestions.unshift('This stock has a relatively high PEG, which may indicate that a stock is overvalued.')
            }else if(stock_price_statistics.value.pegRatio<=0.3){
                temp_suggestions.unshift('This stock has a relatively low PEG, which may indicate that a stock is undervalued.')
            }else{
                temp_suggestions.unshift('This stock has a moderate PEG.')
            }
            labels.unshift('Analysis on Quick Ratio')
            if(stock_price_statistics.value.quickRatio<1){
                temp_suggestions.unshift('WARNING: This company has a quick ratio of less than 1, which indicates that it not be able to fully pay off its current liabilities in the short term.')
            }else{
                temp_suggestions.unshift('This company has a quick ratio of no less than 1, which indicates that it fully equipped with enough assets to be instantly liquidated to pay off its current liabilities.')
            }

            for(var i = 0; i < labels.length ; i++){
                data_suggestions.push({
                    label : labels[i],
                    suggestion : temp_suggestions[i]
                })
            }
            return data_suggestions
        })
        
        const price_change = computed(()=>{
            return stock_basic_info.current_price-diagram_data.value[diagram_data.value.length-1]
        })

        function handleWatchlist(){
            if(proxy.AccountInfo.login==false){
                // console.log(route.query)
                router.push({
                    name: 'Login',
                    params: {
                        from_path : router.currentRoute.value.path
                    },
                    query: route.query
                })
            }else{
                watchlist_button_loading.value = true
                // inwatchlist.value = !inwatchlist.value
                if(inwatchlist.value == true){
                    axios.post(`${proxy.BasicUrl}/watchlist/remove/`,{
                        "account_id" : proxy.AccountInfo.account_id,
                        "stock_id" : stock_basic_info.stock_id
                    }).then(
                        response =>{
                            if(response.data.result=="Success"){
                                inwatchlist.value = false
                                watchlist_button_loading.value = false
                            }
                        },
                        error => {
                            alert("Error with Server!")
                            console.log(error)
                            watchlist_button_loading.value = false
                        }
                    )
                }else{
                    axios.post(`${proxy.BasicUrl}/watchlist/add/`,{
                        "account_id" : proxy.AccountInfo.account_id,
                        "stock_id" : stock_basic_info.stock_id
                    }).then(
                        response =>{
                            if(response.data.result=="Success"){
                                inwatchlist.value = true
                                watchlist_button_loading.value = false
                            }
                        },
                        error => {
                            alert("Error with Server!")
                            console.log(error)
                            watchlist_button_loading.value = false
                        }
                    )
                }
            }
        }

        function checkWatchlist(){
            if(proxy.AccountInfo.login){
                axios.get(`${proxy.BasicUrl}/watchlist/check/?account_id=${proxy.AccountInfo.account_id}&stock_id=${stock_basic_info.stock_id}`).then(
                    response=>{
                        if(response.data.check==true){
                            inwatchlist.value = true
                        }
                    },
                    error=>{
                        console.log(error)
                    }
                )
            }
        }

        function getBasicInfo(){
            const url = `${proxy.BasicUrl}/stock/info/?stock_id=${stock_basic_info.stock_id}`
            axios.get(url).then(
            response => {
                stock_basic_info.company_name = response.data.data[0].company_name
                stock_basic_info.exchange_name = response.data.data[0].exchange_name
                stock_basic_info.currency = response.data.data[0].currency
            },
            error => {
                console.log(error)
            })
        }

        function getCurrentPrice(){
            const url = `${proxy.BasicUrl}/price/current/?stock_id=${stock_basic_info.stock_id}&stock_symbol=${stock_basic_info.stock_symbol}`
            axios.get(url).then(
            response => {
                stock_basic_info.current_price = response.data.data
                stock_basic_info.current_price_time = response.data.dataTime.replace("T", " ")
                getPriceStatistics()
            },
            error => {
                stock_basic_info.current_price = "Error!"
                stock_basic_info.current_price_time = "Error!"
            })
        }

        function getPriceStatistics(){
            const url = `${proxy.BasicUrl}/statistics/current/?stock_id=${stock_basic_info.stock_id}`
            axios.get(url).then(
            response => {
                stock_price_statistics.value = response.data.data
            },
            error => {
                
            })
        }
        
        function getDiagramData(period : number){
            loading.value = true
            const url = `${proxy.BasicUrl}/price/recent/?stock_id=${stock_basic_info.stock_id}&period=${period}&date=True`
            axios.get(url).then(
            response => {
                diagram_data.value = response.data.data
                diagram_date.value = response.data.date
                loading.value = false
            },
            error => {
            }
        )}

        let diagram_config = computed(()=>{
            var temp_diagram_data = [...diagram_data.value]
            var temp_diagram_date = [...diagram_date.value]
            if(predict_show.value){
                temp_diagram_data.push.apply(temp_diagram_data, predict_data.price)
                temp_diagram_date.push.apply(temp_diagram_date, predict_data.date)
            }
            // console.log(temp_diagram_data)
            let d_color = 'green';
            if(temp_diagram_data[temp_diagram_data.length-1] < temp_diagram_data[0]){
                d_color = 'red';
            }
            const min = Math.floor(Math.min(...temp_diagram_data))
            const max = Math.ceil(Math.max(...temp_diagram_data))
            // console.log(min, max)
            const option =  {
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' }
            },
            grid:{
                left: "1%",
                top: "5%",
                right: "4%",
                bottom: "5%",
                containLabel: true
    	    },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: temp_diagram_date,
                axisLabel: {
                    showMaxLabel: true,
                    showMinLabel: true,
                }
            },
            yAxis: {
                type: 'value',
                min: min,
                max: max,
                // scale : true,
                splitNumber : 5
            },
            series: [
            {
                symbol: "none",
                data: temp_diagram_data,
                type: "line",
                areaStyle: {},
                itemStyle : {
    				normal : {
                        color: d_color
                    }
                },
            }
            ],
            }
            return option
        })

        function getPredictData(){
            const url = `${proxy.BasicUrl}/predict/get/?stock_id=${stock_basic_info.stock_id}`
            axios.get(url).then(
            response => {
                response.data.data.forEach((item)=>{
                    predict_data.price.push(item.predict_price)
                    predict_data.date.push(item.price_date + "\n(PREDICT)")
                })
                // console.log(predict_data)
            },
            error => {
            }
        )}

        function getSuggestion(){
            const url = `${proxy.BasicUrl}/suggestion/get/?stock_id=${stock_basic_info.stock_id}`
            axios.get(url).then(
            response => {
                response.data.data.forEach((item)=>{
                    remote_suggestions.push({
                        'label' : 'Suggestion by ' + item.firm,
                        'suggestion': `Time: ${item.rcmd_time.replace('T', ' ')}\n To Grade: ${item.to_grade} \nAction: ${item.action}`
                    })
                })
                // console.log(predict_data)
            },
            error => {
            })
        }

        function getVar(){
            axios.get(`${proxy.BasicUrl}/statistics/var/?stock_id=${stock_basic_info.stock_id}`).then(
                response => {}
            )
        }

        onMounted(()=>{
            getBasicInfo()
            getCurrentPrice()
            getDiagramData(30)
            getPredictData()
            checkWatchlist()
            getSuggestion()
        })
        return{
            stock_basic_info,
            stock_price_statistics,
            diagram_config,
            getDiagramData,
            loading,
            predict_data,
            predict_show,
            Star,
            watchlist_button_text,
            watchlist_button_type,
            handleWatchlist,
            watchlist_button_loading,
            suggestions,
            remote_suggestions,
            price_change
        }
    }

}
</script>

<style>
.el-row {
  margin-bottom: 20px;
}
.centerBox{
    margin: 40px auto;
    width: 90vw;
}
.primaryStockBasicInfo{
    font-size: 30px;
    text-align: left;
    margin-bottom: 10px;
    font-weight: bold;
}
.secondaryStockBasicInfo{
    font-size: 20px;
    text-align: left;
    margin-bottom: 20px;
}
.priceInfo{
    font-size: 50px;
    text-align: left;
    font-weight: bold;
}
.statistics{
    font-size: 16px;
    margin: 50px auto;
    text-align: left;
}
.statistics_left{
    text-align: left;
    margin: 10px 0;
}
.statistics_right{
    text-align: right;
    margin: 10px 0;
    font-weight: bold;
}
.diagram{
    margin-top: 10px;
    margin-bottom: 10px;
    margin-left: 0px;
    margin-right: auto;
    width : auto;
    height: 400px;
}
.multiButtons{
    margin: auto;
}
.demonstration {
  color: var(--el-text-color-secondary);
  font-size: 20px;
}

.el-carousel__item h3 {
    color: #475669;
    opacity: 0.75;
    line-height: 120px;
    margin: 0;
    text-align: center;
}

.el-carousel__item:nth-child(2n) {
  background-color: #99a9bf;
}

.el-carousel__item:nth-child(2n + 1) {
  background-color: #d3dce6;
}
.sug-text{
 white-space: pre-line;
 margin: 0px 0px;
 line-height: 30px;
 width: 100%;
 text-align: left;
}
.red-price-change {
  color: red;
  font-size: 30px;
}
.green-price-change {
  color: green;
  font-size: 30px;
}
</style>