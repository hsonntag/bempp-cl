"""Laplace potential operators."""


def single_layer(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )


def single_layer_gradient(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential gradient operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_potential_gradient",  # Identifier
        [],  # Options
        "laplace_single_layer_gradient",  # Kernel type
        "laplace_single_layer_gradient",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )


def double_layer(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def double_layer_gradient(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential gradient operator."""
    import bempp.api
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator
    from bempp.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_potential_gradient",  # Identifier
        [],  # Options
        "laplace_double_layer_gradient",  # Kernel type
        "laplace_double_layer_gradient",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )
