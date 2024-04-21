#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel void kernel_function(__global REALTYPE *grid,
                                              __global uint* indices,
                                              __global int* normalSigns,
                                              __global REALTYPE *evalPoints,
                                              __global REALTYPE *coefficients,
                                              __constant REALTYPE* quadPoints,
                                              __constant REALTYPE *quadWeights,
                                              __global REALTYPE *globalResult,
					      __global REALTYPE* kernel_parameters) {
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t elementIndex = indices[gid[1]];

  size_t lid = get_local_id(1);
  size_t groupId = get_group_id(1);
  size_t numGroups = get_num_groups(1);

  REALTYPE3 evalGlobalPoint;
  REALTYPE3 surfaceGlobalPoint;

  REALTYPE3 corners[3];
  REALTYPE3 jacobian[2];
  REALTYPE3 normal;
  REALTYPE3 dummy;

  REALTYPE2 point;

  REALTYPE intElement;
  REALTYPE value[NUMBER_OF_SHAPE_FUNCTIONS];

  size_t quadIndex;
  size_t index;

  evalGlobalPoint =
      (REALTYPE3)(evalPoints[3 * gid[0] + 0], evalPoints[3 * gid[0] + 1],
                  evalPoints[3 * gid[0] + 2]);

  // printf("gid0 = %d\n", gid[0]);
  // printf("gid1 = %d\n", gid[1]);
  // printf("lid = %d\n", lid);
  // printf("groupId = %d\n", groupId);
  // printf("numGroups = %d\n", numGroups);
  // printf("novec\n");
#ifndef COMPLEX_RESULT
  __local REALTYPE localResult[WORKGROUP_SIZE][3];
  REALTYPE myResult[3] = {M_ZERO, M_ZERO, M_ZERO};
#else
  printf("complex result\n");
#endif

#ifndef COMPLEX_KERNEL
  REALTYPE kernelValue[3];
#else
  printf("complex kernel\n");
#endif

#ifndef COMPLEX_COEFFICIENTS
  REALTYPE tempResult;
  REALTYPE myCoefficients[NUMBER_OF_SHAPE_FUNCTIONS];
  for (int index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
    myCoefficients[index] =
        coefficients[NUMBER_OF_SHAPE_FUNCTIONS * elementIndex + index];
#else
  printf("complex coefficients\n");
#endif

  getCorners(grid, elementIndex, corners);
  getJacobian(corners, jacobian);
  getNormalAndIntegrationElement(jacobian, &normal, &intElement);

  updateNormals(elementIndex, normalSigns, &normal);

  for (quadIndex = 0; quadIndex < NUMBER_OF_QUAD_POINTS; ++quadIndex) {
    point = (REALTYPE2)(quadPoints[2 * quadIndex], quadPoints[2 * quadIndex + 1]);
    BASIS(SHAPESET, evaluate)(&point, &value[0]);
    surfaceGlobalPoint = getGlobalPoint(corners, &point);
#ifndef COMPLEX_KERNEL
    KERNEL(novec)
    (evalGlobalPoint, surfaceGlobalPoint, dummy, normal, kernel_parameters, kernelValue);
#else
    printf("complex kernel\n");
#endif

#ifndef COMPLEX_COEFFICIENTS
    tempResult = M_ZERO;
    for (index = 0; index < NUMBER_OF_SHAPE_FUNCTIONS; ++index)
      tempResult += myCoefficients[index] * value[index];
    tempResult *= quadWeights[quadIndex];
#ifndef COMPLEX_KERNEL
    myResult[0] += tempResult * kernelValue[0];
    myResult[1] += tempResult * kernelValue[1];
    myResult[2] += tempResult * kernelValue[2];
#else
    printf("complex kernel\n");
#endif
#else
  printf("complex coefficients\n");
#endif
  }

#ifndef COMPLEX_RESULT
  localResult[lid][0] = myResult[0] * intElement;
  localResult[lid][1] = myResult[1] * intElement;
  localResult[lid][2] = myResult[2] * intElement;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    for (index = 1; index < WORKGROUP_SIZE; ++index) {
      localResult[0][0] += localResult[index][0];
      localResult[0][1] += localResult[index][1];
      localResult[0][2] += localResult[index][2];
    }
    globalResult[(3 * gid[0]) * numGroups + groupId] += localResult[0][0];
    globalResult[(3 * gid[0] + 1) * numGroups + groupId] += localResult[0][1];
    globalResult[(3 * gid[0] + 2) * numGroups + groupId] += localResult[0][2];

  }

#else
  printf("complex results\n");
#endif
}
