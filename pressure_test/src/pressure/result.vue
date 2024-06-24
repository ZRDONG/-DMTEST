<script>
import b1 from './button/b1.vue';
import b2 from './button/b2.vue';
import b3 from './button/b3.vue';
import b4 from './button/b4.vue';


export default {
  components: {
    b1,b2,b3,b4
  }
}
</script>

<template>
    <div class="container">
        <el-card class="card" style="max-width: 1500px">
            <div class="content">
                <!-- 头像 -->
                <div class="avatar-container">
                    <el-avatar
                    src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"
                    size="large"
                    />
                </div>
                <!-- 管理员 -->
                <div class="text-container">
                    <span>管理员</span>
                </div>
            </div>
            <div class="button-container">
                <router-link to="/data">
                    <b1>
                        <el-icon class="custom-icon"><List /></el-icon>
                        <span class="button-text">数据设定</span>
                        <el-icon class="bottom-icon"><AlarmClock /></el-icon>
                    </b1>
                </router-link>
                <router-link to="/target">
                    <b2>
                        <el-icon class="custom-icon"><Flag /></el-icon>
                        <span class="button-text">指标收集</span>
                        <el-icon class="bottom-icon"><Connection /></el-icon>
                    </b2>
                </router-link>
                <router-link to="/parameter">
                    <b3>
                        <el-icon class="custom-icon"><TrendCharts /></el-icon>
                        <span class="button-text">参数推荐</span>
                        <el-icon class="bottom-icon"><Star /></el-icon>
                    </b3>
                </router-link>
                <router-link to="/log">
                    <b4>
                        <el-icon class="custom-icon"><Histogram /></el-icon>
                        <span class="button-text">日志管理</span>
                        <el-icon class="bottom-icon"><Share /></el-icon>
                    </b4>
                </router-link>
            </div>
        </el-card>
    </div>
    <br/>
    <el-card style="max-width: 100%;" >
            <router-view></router-view>
    </el-card>
</template>
  
  <style scoped>
  .container {
    display: flex;
    justify-content: center;
  }
  
  .card {
    width: 100%; /* 宽度占满父容器 */
  }
  
  .content {
    display: flex;
    flex-direction: column; /* 垂直方向布局 */
    align-items: flex-start; /* 在垂直方向上居左对齐 */
  }
  
  .avatar-container {
    margin-bottom: 1px; /* 调整头像和文本之间的间距 */
  }

  .text-container {
  margin-left: 5px; /* 调整管理员文字距离左边的间距 */
  }
 
  .button-container {
  margin-top: -85px; /* 调整按钮与内容区域之间的间距 */
  margin-left: 100px;
  display: flex; /* 使用 Flexbox 布局 */
  justify-content: space-between; /* 在容器中平均分配每个项目的空间 */
  align-items: center; /* 垂直居中对齐项目 */
  gap: 10px;/* 间距*/
 }
 
 .custom-icon {
  font-size: 50px; /* 调整图标的大小 */
  margin-right: 20px; /* 调整图标与文字之间的间距 */
  position: relative; /* 允许使用 top 属性 */
  top: 20px; /* 向上移动 3px，根据需要调整这个值 */
}
.button-text {
  vertical-align: middle; /* 设置文字垂直居中对齐 */
  position: relative; /* 允许使用 top 属性 */
  top: -5px; /* 向上移动 3px，根据需要调整这个值 */
  font-size: 35px;
}

.bottom-icon{
    position: relative; /* 允许使用 top 属性 */
    top: 2px; /* 向上移动 3px，根据需要调整这个值 */
    right: -100px;
    font-size: 20px;
}




  </style>
  