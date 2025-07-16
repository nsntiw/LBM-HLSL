#define MAKE_DIRECTX(major, minor) (major * 0x1000 + minor * 0x0100)
#define MAKE_OPENGL(major, minor) (0x10000 + major * 0x01000 + minor * 0x00100)
#define MAKE_VULKAN(major, minor) (0x20000 + major * 0x01000 + minor * 0x00100)
#define MAKE_RESHADE(major, minor, revision) (major * 10000 + minor * 100 + revision)
#define VENDOR_NVIDIA 0x10de
#define VENDOR_AMD 0x1002
#define VENDOR_INTEL 0x8086

#define COMPUTE 0
#if (((__RENDERER__ >= MAKE_DIRECTX(11,0) && __RENDERER__ < MAKE_OPENGL(0,0)) || (__RENDERER__ >= MAKE_OPENGL(4,3))) && __RESHADE__ >= MAKE_RESHADE(4,8,0))
   #define COMPUTE 1
   #define WARPS 32
   #if (__VENDOR__ == VENDOR_AMD)
      #define WARPS 64 //Fill AMD warps
   #endif
	uniform bool IsCompute<
		ui_label = "COMPUTE == 1";
		ui_tooltip = "COMPUTE == 1";
	> = false; //Show that compute shaders are enabled
#endif

struct VSOUT
{
	float4                  vpos        : SV_Position;
   float3                  uv          : TEXCOORD0;   
};
VSOUT MainVS(in uint id : SV_VertexID)
{
   VSOUT o;
   FullscreenTriangleVS(id, o.vpos, o.uv.xy);
   return o;
}
void MainPS(in VSOUT i, out float3 o : SV_Target0) {
   o = float3(0,0,0)
}

//========================================================================//
//For calculating DispatchSize, n = numerator, d = denominator
#define DIVIDE_ROUNDING_UP(n, d) (int(n + d - 1) / int(d))

struct lattice
{
   float a = 0;
   float b = 0;
   float c = 0;
};

#if COMPUTE == 1
   #define NUM_LATTICE 100
   void MainCS(uint3 id : SV_DispatchThreadID, uint3 t_id : SV_GroupThreadID){

   }
#endif


//========================================================================//
technique LBM
<
   ui_label = "LBM";
   ui_tooltip = "";
>
{
   pass    {VertexShader = MainVS;PixelShader = MainPS;}
   #if COMPUTE == 1
	pass
   {
		ComputeShader = BlendPyramidPSWave0CS<GROUP_DIMENSION.x, GROUP_DIMENSION.y>;
		DispatchSizeX = DIVIDE_ROUNDING_UP(BUFFER_WIDTH, GROUP_DIMENSION.x * PX_PER_THREAD.x);
      DispatchSizeY = DIVIDE_ROUNDING_UP(BUFFER_HEIGHT, GROUP_DIMENSION.y * PX_PER_THREAD.y);
   }
   #endif
}