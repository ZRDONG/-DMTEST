<template>
    <button class="big-button">
      <slot></slot>
    </button>
</template>
  
  
  <script>
  export default {
    name: 'BigButton'
  }
  </script>
  
  <style scoped>
  .big-button {
    display: inline-block;
    padding: 0px 20px;
    font-size: 15px;
    color: white;
    background-color: #d44ff1; /* 蓝色底色 */
    border: none;
    border-radius: 10px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  
  
  .big-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background-size: cover;
    background-position: center;
    opacity: 0.1; /* 背景图案透明度 */
    z-index: 1;
  }
  
  .big-button:hover {
    background-color: #7e0c98; /* 悬停时的蓝色变化 */
  }
  </style>
  