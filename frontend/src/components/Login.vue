<template>
  <div>
   <img src="https://lsky.cloud.roymikejiang.tech/i/2022/07/25/62de4b1375f4b.png" class="bg"/>
   <div class="login-page">
      <transition name="fade">
         <div v-if="!registerActive" class="wallpaper-login"></div>
      </transition>
      <div class="wallpaper-register"></div>

      <div class="container">
         <div class="row">
            <div class="col-lg-4 col-md-6 col-sm-8 mx-auto">
               <div v-if="!registerActive" class="card login" v-bind:class="{ error: emptyFields }">
                  <h1>Sign In</h1>
                  <form class="form-group">
                     <el-input v-model="usernameLogin" type="text" class="form-control" placeholder="Username" required></el-input><br/>
                     <el-input v-model="passwordLogin" type="password" class="form-control" placeholder="Password" required></el-input><br/>
                     <el-button class="btn btn-primary" @click="doLogin" size="large">Login</el-button><br/>
                     <p>Don't have an account? <a href="#/login" @click="registerActive = !registerActive, emptyFields = false">Sign up here</a>
                     </p>
                  </form>
               </div>

               <div v-else class="card register" v-bind:class="{ error: emptyFields }">
                  <h1>Sign Up</h1>
                  <form class="form-group">
                     <el-input v-model="usernameReg" type="text" class="form-control" placeholder="Username" required></el-input><br/>
                     <el-input v-model="passwordReg" type="password" class="form-control" placeholder="Password" required></el-input><br/>
                     <el-input v-model="confirmReg" type="password" class="form-control" placeholder="Confirm Password" required></el-input><br/>
                     <el-button class="btn btn-primary" @click="doRegister" size="large">Register</el-button>
                     <p>Already have an account? <a href="#/login" @click="registerActive = !registerActive, emptyFields = false">Sign in here</a>
                     </p>
                  </form>
               </div>
            </div>
         </div>
      </div>
   </div>

</div>
</template>

<script lang="ts" setup>
import {ref, getCurrentInstance} from 'vue'
import axios from 'axios'
import { useCookies } from "vue3-cookies";
import { useRouter, useRoute } from "vue-router";

const router = useRouter();
const route = useRoute();
const { proxy } = getCurrentInstance();
const { cookies } = useCookies();
const registerActive = ref(false)
const usernameLogin = ref("")
const passwordLogin = ref("")
const usernameReg = ref("")
const passwordReg = ref("")
const confirmReg = ref("")
const emptyFields = ref(false)

function doLogin() {
    if (usernameLogin.value == "" || passwordLogin.value == "") {
        emptyFields.value = true;
    } else {
        axios.post(`${proxy.BasicUrl}/account/login/`, {
            "username" : usernameLogin.value,
            "password" : passwordLogin.value
        }).then(
        response => {
            if(response.data.result=="Success"){
               proxy.AccountInfo.login = true
               proxy.AccountInfo.account_id = response.data.data.account_id
               proxy.AccountInfo.username = response.data.data.username
               cookies.set("token", response.data.token, 60*60*24);
               alert("You are now logged in");
               if(route.params.from_path == undefined) {
                  router.push({
                     name: 'Market'
                  }) 
               }else{
                  router.push({
                     path : route.params.from_path,
                     query: route.query
                  })
               }
            }else{
                alert("Wrong Username or Password!");
            } 
        },
        error => {
            alert("Server Error!");
            console.log(error)
        })
    }
} 
function doRegister() {
    if (usernameReg.value === "" || passwordReg.value === "" || confirmReg.value === "" || passwordReg.value != confirmReg.value) {
        emptyFields.value = true;
    } else {
         axios.post(`${proxy.BasicUrl}/account/signup/`, {
            "username" : usernameReg.value,
            "password" : passwordReg.value
        }).then(
         response =>{
            if(response.data.result=="Success"){
               alert("You are now registered");
               registerActive.value = !registerActive.value
               emptyFields.value = false
            }
         },
         error=>{
            alert("Server Error!");
            console.log(error)
         }
        )
    }
}

</script>

<style lang="scss">
img.bg {
    min-height: 95%;
    min-width: 1024px;
    width: 100%;
    height: auto;
    position: fixed;
    top: 59px;
    left: 0;
    opacity: 0.8;
}
p {
   line-height: 1rem;
}

.card {
   padding: 20px;
}

.form-group {
    font-size: 20px;
    input{
        margin: 10px 0;
        font-size: 20px;
    }
    button{
        margin: 10px 0;
        font-size: 20px;
    }
}

.form-control{
    margin-bottom: 10px;
    vertical-align:middle;
    line-height: 100%;
}

.login-page {
   background-color: #ffffff;
   border-radius:15px 40px 15px 40px;
    // margin: auto;
   align-items: center;
   position: absolute;
    position: absolute;
	left: 50%;
	top: 50%;
	transform: translate(-50%,-50%);
   
   .fade-enter-active,
   .fade-leave-active {
  transition: opacity .5s;
}
   .fade-enter,
   .fade-leave-to {
      opacity: 0;
   }

   h1 {
      margin-bottom: 1.5rem;
   }
}
.login-page.dark {
   background-color: #000000;
   border-radius:15px 40px 15px 40px;
    // margin: auto;
   align-items: center;
   position: absolute;
    position: absolute;
	left: 50%;
	top: 50%;
	transform: translate(-50%,-50%);
   
   .fade-enter-active,
   .fade-leave-active {
  transition: opacity .5s;
}
   .fade-enter,
   .fade-leave-to {
      opacity: 0;
   }

   h1 {
      margin-bottom: 1.5rem;
   }
}

.error {
   animation-name: errorShake;
   animation-duration: 0.3s;
}

@keyframes errorShake {
   0% {
      transform: translateX(-25px);
   }
   25% {
      transform: translateX(25px);
   }
   50% {
      transform: translateX(-25px);
   }
   75% {
      transform: translateX(25px);
   }
   100% {
      transform: translateX(0);
   }
}

</style>