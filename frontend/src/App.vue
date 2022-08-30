<template>
  <el-config-provider namespace="ep">
    <BaseHeader />
    <div style="display: flex">
      <Suspense>
        <template #default>
          <router-view></router-view>
        </template>
      </Suspense>
    </div>
  </el-config-provider>
</template>

<script lang="ts" setup>
import {getCurrentInstance} from 'vue'
import axios from 'axios';
const {proxy} = getCurrentInstance()
// axios.defaults.withCredentials = true
axios.get(`api/account/verify/`,{ withCredentials: true }).then(
    response => {
        proxy.AccountInfo.login = true
        proxy.AccountInfo.account_id = response.data.account_info.account_id
        proxy.AccountInfo.username = response.data.account_info.username
    },
    error => {
      console.log(proxy.AccountInfo)
    }
)
</script>

<style>
#app {
  text-align: center;
  color: var(--ep-text-color-primary);
  min-height: 100vh;
}
</style>
