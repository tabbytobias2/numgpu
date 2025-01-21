let initialized: boolean = false;
let pipelineBindLayout: GPUPipelineLayout = undefined;

let bindLayout: GPBindGroupLayout = undefined;
export let device: GPUDevice = undefined;

export let chunkSize = 255;

export const operations: string[] = [
    "+",
    "-",
    "/",
    "*"
];

export async function init() {
    if(navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        device = await adapter.requestDevice();

        bindLayout = await device.createBindGroupLayout({
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type: "storage",
                },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type: "storage",
                },
              },
            ],
        });

        pipelineBindLayout = await device.createPipelineLayout({
            bindGroupLayouts: [bindLayout],
        });

        initialized = true;
    } else {
        throw new Error("Webgpu is not available in this paradigm.");
    }
}

export function uint8ArrayToFloat32Array(uint8Array) {
  const length = uint8Array.length / 4;  // 4 bytes per Float32
  const float32Array = new Float32Array(length);

  // Create a DataView to interpret the Uint8Array as binary data
  const view = new DataView(uint8Array.buffer);

  // Read 4 bytes at a time and convert them to Float32
  for (let i = 0; i < length; i++) {
    float32Array[i] = view.getFloat32(i * 4, true); // 'true' for little-endian, change to 'false' for big-endian
  }

  return float32Array;
}

function checkBuffers(buf1: GPUBuffer, buf2: GPUBuffer) {
  let err: boolean = false;
  if(!(buf1.usage & GPUBufferUsage.STORAGE)) {
    err = true;
  }
  if(!(buf2.usage & GPUBufferUsage.STORAGE)) {
    err = true;
  }
  if(err) {
    throw new Error("Both buffers must have STORAGE permissions.");
  }
  if(buf1.size != buf2.size) {
    throw new Error("Buffers must be the same size.")
  }
}

export async function getBuffer(buf: GPUBuffer): Promise<Float32Array> {
  if (!(buf.usage & GPUBufferUsage.MAP_READ)) {
    if(!(buf.usage & GPUBufferUsage.COPY_SRC)) {
      throw new Error("Buffer must have MAP_READ or COPY_SRC permissions to read it.");
    }
    const outBuf = await device.createBuffer({
      label: "outbuf",
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      size: buf.size,
    });
    const encoder = device.createCommandEncoder();
    await encoder.copyBufferToBuffer(buf, 0, outBuf, 0, buf.size);
    device.queue.submit([encoder.finish()]);
    await outBuf.mapAsync(GPUMapMode.READ);
    const output = uint8ArrayToFloat32Array(new Uint8Array(outBuf.getMappedRange()));
    await outBuf.unmap();
    return output;

  }
  if (buf.mapState !== "unmapped") {
    throw new Error("Another process/thread is currently using this buffer, please unmap it.");
  }

  await buf.mapAsync();
  let output = await buf.getMappedRange();
  await buf.unmap();
  return output;
}

export async function operate(buf1: GPUBuffer, buf2: GPUBuffer, operation: string, times?: number) {
  checkBuffers(buf1, buf2)
  const pipeline = await createOperationPipeline(operation, buf1.size / 4);
  await useOperationPipeline(buf1, buf2, pipeline, times);
}

export async function useOperationPipeline(buf1: GPUBuffer, buf2: GPUBuffer, pipeline: GPUComputePipeline, times?: number) {
  checkBuffers(buf1, buf2);
  const len = buf1.size / 4;

  const encoder = await device.createCommandEncoder();
  const pass = await encoder.beginComputePass();

  const bindGroup = await device.createBindGroup({
    label: "operation bind group",
    layout: bindLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: buf1 },
      },
      {
        binding: 1,
        resource: { buffer: buf2 },
      },
    ],
  });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if(times && times > 0) {
    for(let i = 0; i < times; i++) {
      pass.dispatchWorkgroups(Math.ceil(len / chunkSize), 1, 1);
    }
  } else {
    pass.dispatchWorkgroups(Math.ceil(len / chunkSize), 1, 1);
  }
  pass.end();

  device.queue.submit([encoder.finish()]);
}

export async function createOperationPipeline(operation: string, length: number): Promise<GPUComputePipeline> {
  if(operation.length > 1) {throw new Error("Not an operation.")}

  const code = `
  @group(0) @binding(0) var<storage, read_write> data: array<f32>;
  @group(0) @binding(1) var<storage, read_write> modifier: array<f32>;

  @compute @workgroup_size(${chunkSize}, 1, 1) fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
    if(gi.x > ${length}) {return;};
    data[gi.x] = data[gi.x] ${operation} modifier[gi.x];
  }
  `
  const pipeline = await device.createComputePipeline({
    label: operation,
    layout: pipelineBindLayout,
    compute: {
      module: await device.createShaderModule({
        code: code,
      }),
      entryPoint: "main",
    },
  })
  return pipeline;
}

export async function describeF32Array(array: Float32Array): Promise<{size: number, usage: number}> {
  const descriptor:  {size: number, usage: number} = {
    size: array.length * array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  };

  return descriptor;
}

export async function writeBuffer(buffer: GPUBuffer, array: Float32Array) {
  const arraySize = (await describeF32Array(array)).size;
  if(arraySize != buffer.size) {
    throw new Error("Buffers must be the same size.");
  }
  await device.queue.writeBuffer(buffer, 0, array, 0, buffer.size);
  await device.queue.submit([]);
}

export async function bufferFromF32Array(array: Float32Array): Promise<GPUBuffer> {
  const description = await describeF32Array(array);
  const buffer = await device.createBuffer(description);
  await writeBuffer(buffer, array);
  return buffer;
}
