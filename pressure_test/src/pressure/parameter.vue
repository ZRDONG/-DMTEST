<script setup>
import productService from '@/api/productService';
import { ref, onMounted } from 'vue';
const imageUrl = ref('');
const fetchImage = async () => {
  try {
    const response = await productService.gettpmC(); // 调用后端服务获取图片 URL
    imageUrl.value = "data:image/jpeg;base64," + arrayBufferToBase64(response.data);
    console.log(imageUrl.value);
  } catch (error) {
    console.error("There was an error fetching the image URL!", error);
  }
};
const arrayBufferToBase64 = (buffer) => {
  //第一步，将ArrayBuffer转为二进制字符串
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  //将二进制字符串转为base64字符串
  return window.btoa(binary);
};


onMounted(() => {
  // 组件挂载后执行
  fetchImage(); // 调用 fetchImage 函数
});

</script>

<template>
    <el-card style="max-width: 1500px">
        tpmc变化图        <br/>
      <template v-if="imageUrl">
        <img :src="imageUrl" alt="TPMC变化图" style="width: 50%;height: 50%">
      </template>
      <template v-else>
        Loading...
      </template>
    </el-card>
    <el-table :data="tableData" style="width: 100%">
        <el-table-column prop="date" label="推荐参数" width="180" />
        <el-table-column prop="name" label="变化情况" width="180" />
    </el-table>
</template>

<style>
</style>