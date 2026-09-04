use crate::descriptors::Descriptors;
use crate::{PosUvVertex, Transforms};
use std::mem;

/// How many draws one chunk - and so one render pass - is sized to hold.
///
/// Every draw in a chunk takes an aligned slot in the uniform buffer and, if it
/// carries its own quad, four vertices in the vertex buffer; a chunk is closed
/// and a new render pass started as soon as either runs out. The vertex buffer
/// used to be sized at one *vertex* per object rather than four, which capped a
/// chunk at fifty draws and cost a crowded room a render pass for every fifty
/// objects it drew. Both buffers are now sized for the same number of draws,
/// and that number is what the uniform buffer's 64 KiB binding limit allows at
/// the usual 256-byte alignment.
const ESTIMATED_OBJECTS_PER_CHUNK: u64 = 256;

/// Vertices a draw contributes when it carries its own quad.
const VERTICES_PER_OBJECT: u64 = 4;

pub struct DynamicTransforms {
    pub buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl DynamicTransforms {
    pub fn new(descriptors: &Descriptors) -> Self {
        let buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            // A dynamic offset has to be aligned, so a slot costs a whole
            // stride whatever `Transforms` itself measures.
            size: ((mem::size_of::<Transforms>() as u64)
                .max(descriptors.limits.min_uniform_buffer_offset_alignment as u64)
                * ESTIMATED_OBJECTS_PER_CHUNK)
                .min(descriptors.limits.max_uniform_buffer_binding_size)
                .min(descriptors.limits.max_buffer_size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (mem::size_of::<PosUvVertex>() as u64
                * VERTICES_PER_OBJECT
                * ESTIMATED_OBJECTS_PER_CHUNK)
                .min(descriptors.limits.max_buffer_size),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &descriptors.bind_layouts.transforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(mem::size_of::<Transforms>() as u64),
                    }),
                }],
            });
        Self {
            buffer,
            bind_group,
            vertex_buffer,
        }
    }
}
