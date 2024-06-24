<script>
</script>

<template>
    <el-card style="max-width: 1500px">
        tpmc变化图
        <br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
    </el-card>
    <el-table :data="tableData" style="width: 100%">
        <el-table-column prop="date" label="推荐参数" width="180" />
        <el-table-column prop="name" label="变化情况" width="180" />
    </el-table>
</template>

<style>
</style>