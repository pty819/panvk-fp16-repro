/*
 * panvk fp16 packed-compute correctness reproducer
 *
 * C[M][N] = A[M][K] * B[K][N], M=64 K=8 N=64 (single 8x8-thread workgroup).
 * A/B are plain fp32 buffers; the shader converts inputs to f16vec2 (packed
 * half2) and accumulates in fp16 (GL_EXT_shader_explicit_arithmetic_types_float16).
 *
 * Expected: all 4096 outputs match a CPU reference within 1e-3 absolute.
 * Observed on panvk / Mali-G610 (RK3588, Mesa 26.0.8): a deterministic subset
 * of accumulators comes back 0.0 or wrong. The exact same binary runs PASS on
 * llvmpipe (VK_ICD_FILENAMES=lvp_icd.json), and an fp32-only variant of the
 * same kernel is correct on panvk.
 *
 * Build: gcc -O2 -o repro repro.c -lvulkan   (glslc gemm_fp16.comp -o gemm_fp16.spv)
 * Run:   ./repro
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

#define M 64
#define K 8
#define N 64
#define ABS_TOL 1e-3

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "FAIL %s -> %d line %d\n", #x, r_, __LINE__); exit(2); } } while (0)

static float f32_to_f16_roundtrip(float v) {
    /* exact roundtrip through _Float16; used to mirror the shader's
       fp32 -> half conversion of the inputs in the CPU reference */
    return (float)(_Float16)v;
}

int main(void) {
    FILE *f = fopen("gemm_fp16.spv", "rb");
    if (!f) { fprintf(stderr, "gemm_fp16.spv not found; run: glslc gemm_fp16.comp -o gemm_fp16.spv\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(sz);
    if (fread(code, 1, sz, f) != (size_t)sz) return 2;
    fclose(f);

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "panvk-fp16-repro", .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t nd = 0; CK(vkEnumeratePhysicalDevices(inst, &nd, NULL));
    if (!nd) { fprintf(stderr, "no Vulkan devices\n"); return 2; }
    VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(inst, &nd, pds));
    VkPhysicalDevice pd = VK_NULL_HANDLE; VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    VkPhysicalDeviceDriverProperties drv = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
    p2.pNext = &drv;
    for (uint32_t i = 0; i < nd; i++) {
        vkGetPhysicalDeviceProperties2(pds[i], &p2);
        if (p2.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { pd = pds[i]; break; }
    }
    if (!pd) { vkGetPhysicalDeviceProperties2(pds[0], &p2); pd = pds[0]; }  /* llvmpipe */
    printf("device : %s\n", p2.properties.deviceName);
    printf("driver : %s %s\n", drv.driverName, drv.driverInfo);

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

    VkDescriptorSetLayoutBinding bl[3];
    for (int i = 0; i < 3; i++)
        bl[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dsl = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bl };
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

    /* deterministic inputs, same LCG as original test */
    static float hA[M * K], hB[K * N];
    uint32_t st = 0x12345678u;
    for (long i = 0; i < (long)M * K; i++) { st = st * 1664525u + 1013904223u; hA[i] = ((st >> 8) & 0xffffff) / 16777216.0f - 0.5f; }
    for (long i = 0; i < (long)K * N; i++) { st = st * 1664525u + 1013904223u; hB[i] = ((st >> 8) & 0xffffff) / 16777216.0f - 0.5f; }
    for (long i = 0; i < (long)M * K; i++) hA[i] *= 0.1f;
    for (long i = 0; i < (long)K * N; i++) hB[i] *= 0.1f;

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) { mt = i; break; }

    VkBuffer bufs[3]; void *mapped[3];
    VkDeviceSize sizes[3] = { M * K * 4, K * N * 4, M * N * 4 };
    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool pool; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &pool));
    VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &sl };
    VkDescriptorSet set; CK(vkAllocateDescriptorSets(dev, &ai, &set));
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizes[i], .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
        CK(vkCreateBuffer(dev, &bci, NULL, &bufs[i]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mt };
        VkDeviceMemory mem; CK(vkAllocateMemory(dev, &mai, NULL, &mem));
        CK(vkBindBufferMemory(dev, bufs[i], mem, 0));
        CK(vkMapMemory(dev, mem, 0, mr.size, 0, &mapped[i]));
        VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo){ .buffer = bufs[i], .offset = 0, .range = sizes[i] } };
        vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);
    }
    memcpy(mapped[0], hA, sizeof hA);
    memcpy(mapped[1], hB, sizeof hB);
    memset(mapped[2], 0xAA, sizes[2]);  /* poison: catch unwritten outputs */

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
    vkCmdDispatch(cb, M / 64, N / 64, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    CK(vkQueueSubmit(q, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

    /* CPU reference: inputs converted to half exactly like the shader,
       accumulated in double (abs tol covers fp16 accumulation error) */
    const float *hC = (const float *)mapped[2];
    int bad = 0, shown = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += (double)f32_to_f16_roundtrip(hA[i * K + k]) * f32_to_f16_roundtrip(hB[k * N + j]);
            double got = hC[i * N + j];
            if (fabs(got - ref) > ABS_TOL) {
                if (shown < 10) {
                    printf("  C[%2d][%2d] expected %9.6f  got %9.6f\n", i, j, ref, got);
                    shown++;
                }
                bad++;
            }
        }
    printf("%s: %d/%d wrong (abs tol %g)\n", bad ? "FAIL" : "PASS", bad, M * N, ABS_TOL);
    return bad ? 1 : 0;
}
