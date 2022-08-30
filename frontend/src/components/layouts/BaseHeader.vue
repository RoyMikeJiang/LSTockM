<script lang="ts" setup>
import { toggleDark } from '~/composables';
import { useRouter, useRoute } from 'vue-router'
import {computed, getCurrentInstance} from 'vue'
import { useCookies } from "vue3-cookies";
const { cookies } = useCookies();
const { proxy } = getCurrentInstance()
const router = useRouter()
function RouterGoto(dest: string){
  router.push({
    name : dest
  })
}
function handleLogout(){
  cookies.remove('token')
  router.push({
    name: 'Login',
    params: {
      from_path : router.currentRoute.value.path
    }})
  proxy.AccountInfo.login = false
  proxy.AccountInfo.account_id = 0
  proxy.AccountInfo.username = ''
}

</script>

<template>
  <el-menu mode="horizontal" :ellipsis="false">
    <el-menu-item index="1" @click="RouterGoto('Market')"><text class="logo">LSTockM</text></el-menu-item>
    <el-menu-item index="2" @click="RouterGoto('Market')" class="menu-text">Market</el-menu-item>
    <el-menu-item index="3" @click="RouterGoto('MyStocks')" class="menu-text">My Stocks</el-menu-item>
    <el-menu-item index="4" @click="RouterGoto('PredictRequest')" class="menu-text">Predict Request</el-menu-item>
    <div class="flex-grow" />
    <el-menu-item h="full" index="5" @click="toggleDark()">
      <button class="border-none w-full bg-transparent cursor-pointer" style="height: var(--ep-menu-item-height);">
        <i inline-flex i="dark:ep-moon ep-sunny" />
      </button>
    </el-menu-item>
    <el-sub-menu index="6" :disabled="!proxy.AccountInfo.login">
      <el-menu-item index="6.1" @click="handleLogout" >Log Out</el-menu-item>
    </el-sub-menu>
    
  </el-menu>
</template>

<style scoped>
  .menu-text{
    font-size: 20px;
  }
  .logo{
    font-family: 'Nalieta';
    font-size : 50px;
    position: relative;
    top: 5px;
  }
</style>
