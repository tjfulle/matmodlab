sqa = True

# The materials dictionary defines user developed materials that use the
# standard fortran umat/uhyper/uanisohyper_inv interfaces. This interface is
# recommended to most users who would like to exercise a model written in
# fortran as it skips the steps of having to write a python wrapper and f2py
# interfaces for there model.
materials = {}
materials['neohooke_u'] = {'model': USER, 'response': HYPERELASTIC,
                           'libname': 'neohooke_u',
                           'source_files': [join(MAT_D, 'src/uhyper_neohooke.f90')],
                           'ordering': [XX, YY, ZZ, XY, XZ, YZ]}

# The std_materials is a list of files or directories containing user
# developed materials that interface with Matmodlab by subclassing
# mmd.material.MaterialModel. This interface is intended for models written in
# python or for users who want more control over model response. If this
# interface is used to then call models written in a language other than
# python, the developer will have to insure that their model is callable from
# python. Many of the builtin models are implemented as std_materials and call
# their respective fortran procedures directly.
std_materials = [os.path.join(ROOT_D, 'examples/mat_user.py')]
