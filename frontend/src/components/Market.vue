<template>
  <div class="box">
    <div class="search-box">
      <input type="text" v-model="search_symbol" class="search-left" placeholder="Enter Stock Symbol to Search">
      <button class="search-right" icon="el-icon-search" @click="searchBySymbol(search_symbol)">Search</button>
    </div>
    <el-table :data="tableData" 
    stripe 
    style="width: 90%; margin: 30px auto; font-size: 16px;"
    :row-style="{height: '40px'}"
    max-height="75vh"
    v-loading="loading"
    @row-click="toDetailPage">
      <el-table-column prop="stock_symbol" label="Stock Symbol" min-width="50"/>
      <el-table-column prop="company_name" label="Company" min-width="125"/>
      <el-table-column prop="exchange_name" label="Exchange" min-width="50"/>
      <el-table-column prop="currency" label="Currency" min-width="50"/>
      <el-table-column prop="current_price" label="Current Price" min-width="50"/>
      <el-table-column prop="price_change" label="Price Change" min-width="50"/>
      <el-table-column label="30days Price" width="200" header-align='center'>
        <template #default="scope">
          <div style="height: 100%; width:100%">
            <v-chart style="height: 80%; width:100%" :option="scope.row.config"></v-chart>
          </div>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script lang="ts">

import { color, List, use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components';
import VChart, { THEME_KEY } from 'vue-echarts';
import { onMounted ,ref, reactive, getCurrentInstance, onBeforeUnmount} from 'vue';
import axios from 'axios'
use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
]);
import { useRouter } from 'vue-router'

export default {
  name: 'Market',
  components:{ VChart } ,
  setup(){
    const router = useRouter()
    let loading = ref(false);
    const { proxy } = getCurrentInstance();
    let search_symbol = ref('')
    let tableData = ref([])
    const timeouts = []

    function getPriceData(element, index) {
      tableData.value[index].current_price = 'Loading...'
      tableData.value[index].price_change = 'Loading...'
      tableData.value[index].past_data_ready = false
      tableData.value[index].curr_data_ready = false
      getPastPrice(element['stock_id'], index)
      timeouts.push(setTimeout(function(){
        getCurrentPrice(element['stock_id'], index)
      },3000))
    }
    function calPriceChange(index){
      tableData.value[index].price_change = (tableData.value[index].current_price - tableData.value[index].past_data[tableData.value[index].past_data.length - 1]).toFixed(4)
      tableData.value[index].price_change = (tableData.value[index].price_change<0?"":"+") + String(tableData.value[index].price_change)
    }

    function searchBySymbol(symbol: string) {
      loading.value = true
      tableData.value = []
      const url = `${proxy.BasicUrl}/stock/info/?stock_symbol=${symbol}`
      axios.get(url).then(
        response => {
          loading.value = false
          tableData.value = response.data.data
          tableData.value.forEach((element, index)=>{
            getPriceData(element, index)
          })
        },
        error => {
          console.log(error)
        }
      )
    }
    function randomStock(){
      loading.value = true
      const url = `${proxy.BasicUrl}/stock/random/`
      axios.get(url).then(
        response => {
          loading.value = false
            tableData.value = response.data.data
            tableData.value.forEach((element, index)=>{
              getPriceData(element, index)
            })
        },
        error => {
          console.log(error)
        }
      )
    }
    function getPastPrice(stock_id: number, index: number) {
      const url = `${proxy.BasicUrl}/price/recent/?stock_id=${stock_id}`
      axios.get(url).then(
        response => {
            tableData.value[index].past_data = response.data.data
            tableData.value[index].past_data_ready = true
            tableData.value[index].config = getOption(response.data.data)
            if(tableData.value[index].curr_data_ready == true) {
              calPriceChange(index)
            }
        },
        error => {
          console.log(error)
        }
      )
    }
    function getCurrentPrice(stock_id: number, index: number){
      const url = `${proxy.BasicUrl}/price/current/?stock_id=${stock_id}&stock_symbol=${tableData.value[index].stock_symbol}`
      axios.get(url).then(
        response => {
            tableData.value[index].current_price = response.data.data
            tableData.value[index].curr_data_ready = true
            if(tableData.value[index].past_data_ready == true) {
              calPriceChange(index)
            }
        },
        error => {
          tableData.value[index].current_price = 'ERROR!'
          console.log(error)
        }
      )
    }
    function getOption(price_data) {
      let d_color = 'green';
      if(price_data[price_data.length-1] < price_data[0]){
        d_color = 'red';
      }
      const min = Math.floor(Math.min(...price_data))
      const max = Math.ceil(Math.max(...price_data))
      const option =  {
        grid:{
      x:0,
      y:10,
      x2:0,
      y2:10,
      borderWidth:10
    	  },
        xAxis: {
      type: 'category',
      show:false,
        },
        yAxis: {
          type: 'value',
          show:false,
          splitLine: { show: false },
          min: min,
          max: max
        },
        series: [
          {
            symbol: "none",
            data: price_data,
            type: "line",
            areaStyle: {},
            itemStyle : {
    					normal : {
    					color:d_color,
              }
            },
          }
        ],
        silent: true,
      }
      return option
    }

    function toDetailPage(row,column,event){
      // console.log(router)
      
      router.push({
        name: 'StockDetail',
        query: {
          stock_id : row['stock_id'],
          stock_symbol : row['stock_symbol']
        }
      })
    }

    onMounted(()=>{
      randomStock()
    })
    onBeforeUnmount(()=>{
      // console.log(1)
      timeouts.forEach(element =>{
        // console.log(element)
        clearTimeout(element)
      })
    })
    return{
      loading,
      tableData,
      getOption,
      searchBySymbol,
      search_symbol,
      toDetailPage
    }
  }
}
</script>

<style scoped>
 
 .box{
     text-align:center;
     width: 98vw;
     margin-top:20px;
 }
 .search-box{
     margin: 10px auto; 
     width: 750px;
 }
.search-left{
    text-indent: 20px;
     width:80%;
    height:50px;
    border:rgb(77, 166, 77) 1px solid;
    margin-top:20px;
    border-bottom-left-radius:25px;
    border-top-left-radius:25px;
    outline:none;
    font-size: 20px;
}
.search-right{
    width:19%;
    height:54.2px;
    background:rgb(77, 166, 77);
    color: #fff;
    border:none;
    margin-top:20px;
    border-bottom-right-radius:25px;
    border-top-right-radius:25px;
    outline:none;
    font-size: 20px;
    cursor:pointer;
}
</style>
