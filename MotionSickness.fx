uniform bool InvertMode <
	ui_label = "InvertMode";
	ui_tooltip =
		"a";
> = false;

uniform bool IntertBlend <
	ui_label = "IntertBlend";
	ui_tooltip =
		"a";
> = false;

uniform int Red <
	ui_min = 0; ui_max = 255;
	ui_tooltip = "Red";
	ui_category = "Colour";
> = 255;
uniform int Green <
	ui_min = 0; ui_max = 255;
	ui_tooltip = "Green";
	ui_category = "Colour";
> = 0;
uniform int Blue <
	ui_min = 0; ui_max = 255;
	ui_tooltip = "Blue";
	ui_category = "Colour";
> = 0;

#include "ReShade.fxh"

//========================================================================//
texture BackBuffer : COLOR;
sampler sBackBuffer {Texture = BackBuffer;};

//uniform colour = float3(Red, Green, Blue);

void MainPS(float4 vpos : SV_POSITION, float2 texcoord : TEXCOORD, out float3 o : SV_Target0)
{
	//o = float4(tex2Dfetch(sBackBuffer, texcoord * float2(BUFFER_WIDTH, BUFFER_HEIGHT)).rgb, 1);
    o = tex2D(sBackBuffer, texcoord).rgb;
    if (texcoord.r % 4==0){
        o=float3(1.0,0.0,0.0);
    }

    //RENDER_WIDTH
    //RENDER_HEIGHT
    //BUFFER_SCREEN_SIZE
    
}


technique MotionSickness
{
    pass{VertexShader = PostProcessVS; PixelShader  = MainPS;}
}