<script setup>
import {
    Delete,
    Search
} from '@element-plus/icons-vue'
import { ref, onMounted } from 'vue';
import productService from '@/api/productService';


const Parameter = ref([])
let ParameterPage = ref([])

//分页条数据模型
const pageNum = ref(1)//当前页
const pageSize = ref(3)//每页条数
let total = ref()

const PageParameter = () => {
    try {
    console.log(Parameter.value)
    // await fetchParameter()
    ParameterPage.value = Parameter.value.slice((pageNum.value - 1) * pageSize.value, pageNum.value * pageSize.value);
    console.log(ParameterPage.value);
  } catch (error) {
    console.error('There was an error Paging the parameter!', error);
  }
}

const fetchParameter = async () => {
  try {
    const response = await productService.getParameter();
    Parameter.value = response.data;
    total.value = Parameter.value.length
    console.log(Parameter.value)
    console.log(total.value)
    PageParameter()
  } catch (error) {
    console.error('There was an error fetching the parameter!', error);
  }
};

onMounted(() => {
  fetchParameter()
  console.log(Parameter.value)
});



//用户搜索时选中的分类type
const parameterType=ref('')


//当每页条数发生了变化，调用此函数
const onSizeChange = (size) => {
    pageSize.value = size
    pageNum.value = 1
    PageParameter()
}
//当前页码发生变化，调用此函数
const onCurrentChange = (num) => {
    pageNum.value = num
    PageParameter()
}


//控制添加分类弹窗
const dialogVisible = ref(false)

//添加分类数据模型
const parameterData = ref({
    "parameterName": "",
    "parameterType": "",
    "parameterDefalt": "",
    "remark": ""
})
//添加分类表单校验
const rules = {
    stationID: [
        { required: true, message: '请输入测站编号', trigger: 'blur' },
    ],
    stationName: [
        { required: true, message: '请输入测站名称', trigger: 'blur' },
    ],
    stationType: [
        { required: true, message: '请输入测站类型', trigger: 'blur' },
    ]
}

import { ElMessage } from 'element-plus'
//调用接口，添加测站
const addParameter = async ()=>{
    /*let result = await stationCategoryAddService(stationData.value);*/
    ElMessage.success(result.message? result.message:'添加成功')
    //隐藏弹窗
    dialogVisible.value = false;
    //再次访问后台接口，查询所有分类
    /*getStation()*/
}


import { ElMessageBox } from 'element-plus'
const deleteCategory = (row) => {
    ElMessageBox.confirm(
        '你确认删除该分类信息吗？',
        '温馨提示',
        {
            confirmButtonText: '确认',
            cancelButtonText: '取消',
            type: 'warning',
        }
    )
        .then(async() => {
            //用户点击了确认
            /*let result = await stationCategoryDeleteService(row.stationID)*/
            let result=true;
            ElMessage.success(result.message?result.message:'删除成功')
            //再次调用getAllCategory，获取所有文章分类
            /*getStation()*/
        })
        .catch(() => {
            //用户点击了取消
            ElMessage({
                type: 'info',
                message: '取消删除',
            })
        })
}

const drawer = ref(false)

</script>
<template>
    <el-card class="page-container">
        <template #header>
            <div class="header">
                
                <div class="extra">
                    <el-button type="primary" plain size="large">开始压测</el-button>
                    <el-button type="info" plain size="large">中断压测</el-button>
                    <el-button type="success" plain size="large">继续压测</el-button>
                </div>
            </div>
        </template>
         <!-- 搜索表单 -->
         <el-form inline>
            <el-form-item label="参数类型：" >
            <el-select placeholder="请选择" v-model="parameterType" style="width: 150px;">
                <el-option label="内存" value="内存"></el-option>
                <el-option label="缓存" value="缓存"></el-option>
                <el-option label="日志" value="日志"></el-option>
                <el-option label="benchmarksql" value="benchmarksql"></el-option>
            </el-select>
        </el-form-item>

            <el-form-item>
                <el-button type="primary" @click="getStation">搜索</el-button>
                <el-button @click="stationType=''">重置</el-button>
            </el-form-item>
        </el-form>
        <el-table :data="ParameterPage" style="width: 100%" :header-cell-style="{textAlign: 'center'}" :cell-style="{textAlign: 'center'}" stripe>
            <el-table-column label="选择" >
                <el-checkbox label="" value="Value A" />
            </el-table-column>
            <el-table-column label="参数名称" prop="DBNAME" ></el-table-column>
            <el-table-column label="参数类型" prop="DBTYPE" ></el-table-column>
            <el-table-column label="参数默认值" prop="DBDEFALT" ></el-table-column>
            <el-table-column label="参数取值范围" prop="DBRANGE" ></el-table-column>
            <el-table-column label="操作" width="100"  >
                <template #default="{ row }">
                    <el-button :icon="Delete" circle plain type="danger" @click="deleteCategory(row)"></el-button>
                </template>
            </el-table-column>
            <el-table-column label="备注" prop="NOTE">
<!--                <el-popover-->
<!--                    :width="300"-->
<!--                    popper-style="box-shadow: rgb(14 18 22 / 35%) 0px 10px 38px -10px, rgb(14 18 22 / 20%) 0px 10px 20px -15px; padding: 20px;"-->
<!--                    >-->
<!--                    <template #reference>-->
<!--                        <el-avatar src="https://avatars.githubusercontent.com/u/72015883?v=4" />-->
<!--                    </template>-->
<!--&lt;!&ndash;                    <template #default>&ndash;&gt;-->
<!--                        <div-->
<!--                        class="demo-rich-conent"-->
<!--                        style="display: flex; gap: 16px; flex-direction: column"-->
<!--                        >-->
<!--                        <el-avatar-->
<!--                            :size="60"-->
<!--                            src="https://avatars.githubusercontent.com/u/72015883?v=4"-->
<!--                            style="margin-bottom: 8px"-->
<!--                        />-->
<!--                        </div>-->
<!--                      <span></span>-->
<!--                    </template>-->
<!--                </el-popover>-->
            </el-table-column>
            <template #empty>
                <el-empty description="没有数据" />
            </template>
        </el-table>
        <el-button class="mt-4" style="width: 100%" dialogVisible="true" >
            添加参数
        </el-button>

        <!-- 分页条 -->
        <el-pagination v-model:current-page="pageNum" v-model:page-size="pageSize" :page-sizes="[3, 5 ,10, 15]"
            layout="jumper, total, sizes, prev, pager, next" background :total="total" @size-change="onSizeChange"
            @current-change="onCurrentChange"   style="margin-top: 20px; justify-content: flex-end" />
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