/*
 * Diagnostic host: runs the kernel once, then classifies each wrong C
 * element by structural match:
 *   zero     : got == 0
 *   miss_k   : got == ref - p_k  (the k-th product term missing)
 *   only_k   : got == p_k        (only one product accumulated)
 *   double_k : got == ref + p_k
 *   perm     : got equals ref of another C element (permutation)
 * For D (chk mode): zero / perm / partial-chain matches.
 * usage: diag <spv> <chk|nochk> <tile> <seed>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

#define M 64
#define K 8
#define N 64

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "FAIL %s -> %d line %d\n", #x, r_, __LINE__); exit(2); } } while (0)

static float h2(float v) { return (float)(_Float16)v; }

static float hA[M * K], hB[K * N], refC[M * N], refD[64];

static void gen(int seed) {
    uint32_t st = 0x12345678u ^ (seed * 0x9e3779b9u);
    for (long i = 0; i < (long)M * K; i++) { st = st * 1664525u + 1013904223u; hA[i] = (((st >> 8) & 0xffffff) / 16777216.0f - 0.5f) * 0.1f; }
    for (long i = 0; i < (long)K * N; i++) { st = st * 1664525u + 1013904223u; hB[i] = (((st >> 8) & 0xffffff) / 16777216.0f - 0.5f) * 0.1f; }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double r = 0;
            for (int k = 0; k < K; k++) r += (double)h2(hA[i * K + k]) * h2(hB[k * N + j]);
            refC[i * N + j] = (float)r;
        }
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s spv chk|nochk tile seed\n", argv[0]); return 2; }
    int chkmode = !strcmp(argv[2], "chk");
    int tile = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int wg = 64 / (8 * tile);
    gen(seed);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("spv"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(sz);
    if (fread(code, 1, sz, f) != (size_t)sz) return 2;
    fclose(f);

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nd = 0; CK(vkEnumeratePhysicalDevices(inst, &nd, NULL));
    VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(inst, &nd, pds));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    for (uint32_t i = 0; i < nd; i++) {
        vkGetPhysicalDeviceProperties2(pds[i], &p2);
        if (p2.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { pd = pds[i]; break; }
    }
    if (!pd) { vkGetPhysicalDeviceProperties2(pds[0], &p2); pd = pds[0]; }
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qfi = 0;
    for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    float pr = 1.f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfi, .queueCount = 1, .pQueuePriorities = &pr };
    VkPhysicalDeviceShaderFloat16Int8Features f16i8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16 = VK_TRUE };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f16i8, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    VkDevice dev; CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);

    int nbind = chkmode ? 4 : 3;
    VkDescriptorSetLayoutBinding bl[4];
    for (int i = 0; i < nbind; i++)
        bl[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dsl = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = bl };
    VkDescriptorSetLayout sl; CK(vkCreateDescriptorSetLayout(dev, &dsl, NULL, &sl));
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &sl };
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo sm = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)sz, .pCode = code };
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, NULL, &mod));
    int specData[2] = { K, N };
    VkSpecializationMapEntry me[2] = {
        { .constantID = 0, .offset = 0, .size = 4 },
        { .constantID = 1, .offset = 4, .size = 4 } };
    VkSpecializationInfo spec = { .mapEntryCount = 2, .pMapEntries = me, .dataSize = 8, .pData = specData };
    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main",
                   .pSpecializationInfo = &spec },
        .layout = pl };
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        fprintf(stderr, "  memtype %u: heap %u flags=%c%c%c%c\n", i, mp.memoryTypes[i].heapIndex,
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 'D' : '-',
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? 'V' : '-',
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 'C' : '-',
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 'K' : '-');
    uint32_t mt = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) { mt = i; break; }
    if (getenv("COHERENT"))
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool pool; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &pool));
    VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &sl };
    VkDescriptorSet set; CK(vkAllocateDescriptorSets(dev, &ai, &set));

    VkDeviceSize sizes[4] = { M * K * 4, K * N * 4, M * N * 4, 1024 * 4 };
    void *mapped[4];
    VkDeviceMemory mems[4] = {0};
    for (int i = 0; i < nbind; i++) {
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizes[i], .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
        VkBuffer buf; CK(vkCreateBuffer(dev, &bci, NULL, &buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mt };
        VkDeviceMemory mem; CK(vkAllocateMemory(dev, &mai, NULL, &mem));
        CK(vkBindBufferMemory(dev, buf, mem, 0));
        CK(vkMapMemory(dev, mem, 0, mr.size, 0, &mapped[i]));
        mems[i] = mem;
        VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo){ .buffer = buf, .offset = 0, .range = sizes[i] } };
        vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);
    }
    memcpy(mapped[0], hA, sizeof hA);
    memcpy(mapped[1], hB, sizeof hB);
    memset(mapped[2], 0xAA, sizes[2]);
    if (chkmode) memset(mapped[3], 0xAA, sizes[3]);
    if (getenv("FLUSH")) {
        VkMappedMemoryRange mr[4];
        int nm = 0;
        for (int i = 0; i < nbind; i++)
            mr[nm++] = (VkMappedMemoryRange){ .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = mems[i], .offset = 0, .size = VK_WHOLE_SIZE };
        CK(vkFlushMappedMemoryRanges(dev, nm, mr));
    }

    VkCommandPoolCreateInfo cp = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfi };
    VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cp, NULL, &cpool));
    VkCommandBufferAllocateInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbi, &cb));
    VkFenceCreateInfo fc = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CK(vkCreateFence(dev, &fc, NULL, &fence));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cb, (uint32_t)wg, (uint32_t)wg, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    CK(vkQueueSubmit(q, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

    if (getenv("FLUSH")) {
        VkMappedMemoryRange mr[4];
        int nm = 0;
        for (int i = 0; i < nbind; i++)
            mr[nm++] = (VkMappedMemoryRange){ .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = mems[i], .offset = 0, .size = VK_WHOLE_SIZE };
        CK(vkInvalidateMappedMemoryRanges(dev, nm, mr));
    }
    /* ---- classify C ---- */
    const float *hC = (const float *)mapped[2];
    int wrong = 0, czero = 0, missk[K] = {0}, onlyk[K] = {0}, doublek[K] = {0}, permhit = 0, nomatch = 0;
    float p[M][N][K];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                p[i][j][k] = h2(hA[i * K + k]) * h2(hB[k * N + j]);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float got = hC[i * N + j], ref = refC[i * N + j];
            if (fabs(got - ref) <= 1e-3) continue;
            wrong++;
            if (got == 0.0f) { czero++; continue; }
            int matched = 0;
            for (int k = 0; k < K && !matched; k++) {
                if (fabs(got - (ref - p[i][j][k])) < 1e-4) { missk[k]++; matched = 1; }
                else if (fabs(got - p[i][j][k]) < 1e-4) { onlyk[k]++; matched = 1; }
                else if (fabs(got - (ref + p[i][j][k])) < 1e-4) { doublek[k]++; matched = 1; }
            }
            if (matched) continue;
            for (long t = 0; t < (long)M * N; t++)
                if (hC[t] == got && fabs(refC[t] - ref) > 1e-3) { permhit++; matched = 1; break; }
            if (!matched) {
                if (nomatch < 6)
                    printf("  no-match: C[%d][%d] ref=%.6f got=%.6f (ratio %.4f, diff %.6f)\n",
                           i, j, ref, got, ref != 0 ? got / ref : 0, got - ref);
                nomatch++;
            }
        }
    printf("C: wrong=%d zero=%d | miss_k=[%d,%d,%d,%d,%d,%d,%d,%d] only_k=[%d,%d,%d,%d,%d,%d,%d,%d] dbl_k=[%d,%d,%d,%d,%d,%d,%d,%d] perm=%d nomatch=%d\n",
           wrong, czero,
           missk[0],missk[1],missk[2],missk[3],missk[4],missk[5],missk[6],missk[7],
           onlyk[0],onlyk[1],onlyk[2],onlyk[3],onlyk[4],onlyk[5],onlyk[6],onlyk[7],
           doublek[0],doublek[1],doublek[2],doublek[3],doublek[4],doublek[5],doublek[6],doublek[7],
           permhit, nomatch);

    /* ---- classify D ---- */
    if (chkmode) {
        const float *hD = (const float *)mapped[3];
        int dwrong = 0, dzero = 0, dperm = 0, dnomatch = 0;
        for (int t = 0; t < 64; t++) {
            float got = hD[t];
            /* expected serial fp16 chain for thread t */
            _Float16 chk = 0;
            int x = t % 8;
            int r0 = x * 8;
            for (int i = 0; i < 8; i++)
                for (int kk = 0; kk < 8; kk++) {
                    _Float16 h = hA[(r0 + i) * K + kk];
                    chk = chk + (h + h);
                }
            float ref = (float)chk;
            refD[t] = ref;
            if (fabs(got - ref) <= 0.05) continue;
            dwrong++;
            if (got == 0.0f) { dzero++; continue; }
            int matched = 0;
            for (int t2 = 0; t2 < 64; t2++) {
                _Float16 c2 = 0;
                int x2 = t2 % 8, r02 = x2 * 8;
                for (int i = 0; i < 8; i++)
                    for (int kk = 0; kk < 8; kk++) {
                        _Float16 h = hA[(r02 + i) * K + kk];
                        c2 = c2 + (h + h);
                    }
                if (fabs(got - (float)c2) < 1e-3) { dperm++; matched = 1; break; }
            }
            if (!matched) {
                if (dnomatch < 6)
                    printf("  D no-match: tid=%d ref=%.6f got=%.6f (ratio %.4f)\n", t, ref, got, ref != 0 ? got / ref : 0);
                dnomatch++;
            }
        }
        printf("D: wrong=%d zero=%d perm=%d nomatch=%d\n", dwrong, dzero, dperm, dnomatch);
    }
    return 0;
}
