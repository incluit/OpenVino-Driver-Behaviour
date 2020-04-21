# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# check for variable set by setupvars.sh in the SDK, need it to find models
: ${InferenceEngine_DIR:?"Must source the setupvars.sh in the SDK to set InferenceEngine_DIR"}

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

PROJECT_PATH=$parent_path/../
modelDir=$parent_path/../models

### OPENVINO VERSION TO COMPILE ###
# 2019 = 2019
# 2020 = 2020 R1
if (echo $INTEL_CVSDK_DIR | grep -q "openvino_2020"); 
then export OPENVINO_VER=2020
else 
    if (echo $INTEL_CVSDK_DIR | grep -q "openvino_2019"); 
    then export OPENVINO_VER=2019
    else
            export OPENVINO_VER=2019
    fi
fi

# Face Detection
modName=face-detection-adas-0001
export face116=$modelDir/FP16/$modName.xml
export face132=$modelDir/FP32/$modName.xml

modName=face-detection-retail-0004
export face216=$modelDir/FP16/$modName.xml
export face232=$modelDir/FP32/$modName.xml

modName=facial-landmarks-35-adas-0001
export lm116=$modelDir/FP16/$modName.xml
export lm132=$modelDir/FP32/$modName.xml

modName=landmarks-regression-retail-0009
export lm216=$modelDir/FP16/$modName.xml
export lm232=$modelDir/FP32/$modName.xml

modName=person-detection-action-recognition-classroom-0003
export pda16=$modelDir/FP16/$modName.xml
export pda32=$modelDir/FP32/$modName.xml

modName=person-reidentification-retail-0079
export pr116=$modelDir/FP16/$modName.xml
export pr132=$modelDir/FP32/$modName.xml

modName=emotions-recognition-retail-0003
export em16=$modelDir/FP16/$modName.xml
export em32=$modelDir/FP32/$modName.xml

modName=head-pose-estimation-adas-0001
export hp16=$modelDir/FP16/$modName.xml
export hp32=$modelDir/FP32/$modName.xml

modName=face-reidentification-retail-0095
export reid16=$modelDir/FP16/$modName.xml
export reid32=$modelDir/FP32/$modName.xml

modName=frozen_yolo_v3
export yolo16=$parent_path/../data/$modName.xml

export PROJECT_PATH=${PROJECT_PATH}
