uniform int blend_mode <
	ui_type = "combo";
	ui_items = "Invert\0Invert With Blend\0Blend\0";
	ui_category = "Blend Mode";
> = 0;

uniform int Red <
	ui_min = 0; ui_max = 255;
	ui_category = "Colour";
> = 255;
uniform int Green <
	ui_min = 0; ui_max = 255;
	ui_category = "Colour";
> = 0;
uniform int Blue <
	ui_min = 0; ui_max = 255;
	ui_category = "Colour";
> = 0;

uniform int num_x <
	ui_min = 0; ui_max = 255;
	ui_category = "A";
> = 10;

uniform int num_y <
	ui_min = 0; ui_max = 255;
	ui_category = "A";
> = 10;

uniform int size <
	ui_min = 0; ui_max = 255;
	ui_category = "A";
> = 30;

uniform bool Circular <
	ui_category = "A";
> = false;

#include "ReShade.fxh"

//========================================================================//
texture BackBuffer : COLOR;
sampler sBackBuffer {Texture = BackBuffer;};

//Future: Use Mask texture lookup instead of calculating per pixel. Add Mask read option so users can create custom masks
//texture Mask {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8;};
//sampler sMask {Texture = Mask;};

//BUFFER_WIDTH, BUFFER_HEIGHT, BUFFER_SCREEN_SIZE
void MainPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD, out float3 o : SV_Target0)
{
	o = tex2D(sBackBuffer, texcoord).rgb;
	const float3 blend_colour = float3(Red, Green, Blue) / 255.0;
	const uint2 interval_xy = uint2(BUFFER_WIDTH, BUFFER_HEIGHT) / uint2(num_x + 1, num_y + 1); //num(bifurcations)+1=intervals

	float2 shifted_vpos = vpos.xy+size/2; //shift vpos by half size since % oprator work on the top left corner

	//What an ugly solution
	if (all(shifted_vpos >= interval_xy) && all(shifted_vpos <= interval_xy*uint2(num_x+1, num_y+1)) && all(shifted_vpos % interval_xy < size)){
		switch(blend_mode){
			case 0:
				o = -o;
				break;
			case 1:
				//0=Invert, 0.5 Invert and blend, 1, blend
				o = lerp(-o, blend_colour, 0.5);
				break;
			case 2:
				o = blend_colour;
				break;
		}
	}
}


technique MotionSickness
{
	pass{VertexShader = PostProcessVS; PixelShader  = MainPS;}
}