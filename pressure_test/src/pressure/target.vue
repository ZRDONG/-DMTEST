<script setup>
import {
    Star
} from '@element-plus/icons-vue'
import { ref } from 'vue'

const radio1=ref('1')

const target1 = ref([
    {
        "time": "2113020101",
        "name": "1",
        "situation": "1",
    }
])

const target2 = ref([
    {
        "time": "666666666",
        "name": "1",
        "situation": "1",
    }
])


</script>
<template>
    <el-radio-group v-model="radio1">
        <el-radio value="1" size="large" border>系统参数</el-radio>
        <el-radio value="2" size="large" border>数据库参数</el-radio>
    </el-radio-group>
    <br/><br/>
    <el-card>
        <el-table v-if="radio1==='1'" :data="target1" style="width: 100%" :header-cell-style="{textAlign: 'center'}" :cell-style="{textAlign: 'center'}" stripe>
            <el-table-column label="收集日期" prop="time" ></el-table-column>
            <el-table-column label="指标名称" prop="name" ></el-table-column>
            <el-table-column label="收集情况" prop="name" >
                <el-radio-group v-model="radio1" size="large">
                    <el-radio-button label="一轮" value="一轮" />
                    <el-radio-button label="十轮" value="十轮" />
                    <el-radio-button label="五十轮" value="五十轮" />
                    <el-radio-button label="一百论" value="一百轮" />
                </el-radio-group>
            </el-table-column>      
            <template #empty>
                <el-empty description="没有数据" />
            </template>
        </el-table>
        <el-table v-if="radio1==='2'" :data="target2" style="width: 100%" :header-cell-style="{textAlign: 'center'}" :cell-style="{textAlign: 'center'}" stripe>
            <el-table-column label="收集日期" prop="time" ></el-table-column>
            <el-table-column label="指标名称" prop="name" ></el-table-column>
            <el-table-column label="收集情况" prop="name" >
                <el-radio-group v-model="radio1" size="large">
                    <el-radio-button label="一轮" value="一轮" />
                    <el-radio-button label="十轮" value="十轮" />
                    <el-radio-button label="五十轮" value="五十轮" />
                    <el-radio-button label="一百论" value="一百轮" />
                </el-radio-group>
            </el-table-column>     
            <template #empty>
                <el-empty description="没有数据" />
            </template>
        </el-table>
    </el-card>
</template>

<style lang="scss" scoped>

.page-container {
    min-height: 100%;
    box-sizing: border-box;

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
}


</style>