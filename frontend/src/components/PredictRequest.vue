<template>
<div class="box">
<el-row :gutter="20">
  <el-col :span="14"></el-col>
  <el-col :span="6">
    <el-input style="height: 40px; font-size: 18px" v-model="request_form.stock_symbol" placeholder="Input Stock Symbol for Request" round></el-input>
  </el-col>
  <el-col :span="2">
    <el-button style="height: 40px; font-size: 18px" @click="handleRequest()">Request</el-button>
  </el-col>
  <el-col :span="2"></el-col>
</el-row>
<el-row>
  <div class="box">
    <el-table :data="tableData" 
    stripe 
    style="width: 90%; margin: 30px auto; font-size: 16px;"
    :row-style="{height: '60px'}"
    max-height="75vh"
    v-loading="loading"
    @row-click="toDetailPage">
      <el-table-column prop="stock_symbol" label="Stock Symbol" min-width="100"/>
      <el-table-column prop="request_time" label="Request Time" min-width="100"/>
      <el-table-column prop="status" label="Status" min-width="100"/>
      <el-table-column prop="finish_time" label="Finish Time" min-width="100"/>
    </el-table>
  </div>
</el-row>
</div>
</template>

<script lang="ts" setup>
import {ref, reactive, getCurrentInstance, onMounted, computed} from 'vue';
import {useRouter} from 'vue-router'
import axios  from 'axios';

const router = useRouter()
let loading = ref(true);
const { proxy } = getCurrentInstance();
const response_data = ref([])
const tableData = computed(()=>{
  var data = []
  response_data.value.forEach((element)=>{
    data.push({
      stock_symbol : element.stock_symbol,
      request_time : element.request_time!=null ? element.request_time.replace("T", " ") : "",
      status : element.status,
      finish_time : element.finish_time!=null ? element.finish_time.replace("T", " ") : "",
      stock_id : element.stock_id
    })
  })
  return data;
})

const request_form = reactive({
  account_id : proxy.AccountInfo.account_id,
  stock_symbol : ""
})

if(proxy.AccountInfo.login==false){
  router.push({
    name: 'Login',
    params: {
      from_path : router.currentRoute.value.path
    }
  })
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

function getRequestList(){
  axios.get(`${proxy.BasicUrl}/predict/request/get/?account_id=${proxy.AccountInfo.account_id}`).then(
    response => {
      response_data.value = response.data.data
      loading.value = false
    },
    error => {
      console.log(error)
    }
  )
}

function handleRequest(){
  axios.post(`${proxy.BasicUrl}/predict/request/`, request_form).then(
    response => {
      if(response.data.result=="Success"){
        setTimeout(function(){getRequestList()},2000)
      }else{
        alert("Error with Request!")
      }
    },
    error => {
      console.log(error)
      alert("Error with Request!")
    }
  )
}

onMounted(()=>{
  getRequestList()
})

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