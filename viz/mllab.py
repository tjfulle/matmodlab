from enthought.traits.api import (HasStrictTraits, List, Instance, String,
                                  BaseInt, Int, Float, Bool, Property, Button,
                                  Constant, Enum, Tuple, File, on_trait_change)
from enthought.traits.ui.api import (View, Label, Group, HGroup, VGroup, Item,
                                     UItem, TabularEditor, InstanceEditor,
                                     ListEditor, Spring, Action, Handler)
from enthought.traits.ui.tabular_adapter import TabularAdapter

from viz.mdldat import MMLModel
from viz.runner import ModelRunner, IModelRunnerCallbacks
from viz.models import load_models


class MMLInputStringPreview(HasStrictTraits):
    class ISPHandler(Handler):
        def _run(self, info):
            preview = info.ui.context['object']
            preview.runner.RunInputString(preview.inpstr, preview.model)

        def _close(self, info):
            info.ui.dispose()

    inpstr = String
    runner = Instance(ModelRunner)
    model = Instance(MMLModel)

    trait_view = View(
        VGroup(Item('inpstr', style='custom', show_label=False)),
        buttons=[Action(name='Close', action='_close'),
                 Action(name='Run', action='_run')],
        handler=ISPHandler(), width=800, height=600, resizable=True)


class MMLMaterialModelSelector(HasStrictTraits):
    model_type = Enum('Mechanical', 'eos', 'any')
    models = List(Instance(MMLModel))
    selected_model = Instance(MMLModel)
    runid = String
    auto_generated = Bool(True)
    none_constant = Constant("None")
    show_button = Button("Show Input File")
    run_button = Button("Run Material Model")
    rerun = Bool(False)
    supplied_data = List(Tuple(String, File))
    callbacks = Instance(IModelRunnerCallbacks)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        if self.models is None or len(self.models) < 1:
            self.models = load_models()
            if len(self.models) > 0:
                self.selected_model = self.models[0]
            for model in self.models:
                model.supplied_data = self.supplied_data

    def _runid_changed(self, info):
        self.auto_generated = False

    @on_trait_change('selected_model.selected_material')
    def update_sim_name(self):
        if self.auto_generated:
            if (self.selected_model is not None and
                self.selected_model.selected_material is not None):
                runid = "{0}_{1}".format(self.selected_model.name,
                                         self.selected_model.selected_material.name)
                self.runid = "_".join(runid.split())

            # A trick to reset the flag, since _runid_changed() is
            # called first
            self.auto_generated = True

    def _run_button_fired(self, event):
        runner = ModelRunner(runid=self.runid,
                             material_models=[self.selected_model],
                             callbacks=self.callbacks)
        runner.RunModels()

    def _show_button_fired(self, event):
        runner = ModelRunner(runid=self.runid,
                             material_models=[self.selected_model],
                             callbacks=self.callbacks)
        inpstr = runner.CreateModelInputString(self.selected_model)
        preview = MMLInputStringPreview(
            inpstr=inpstr, runner=runner, model=self.selected_model)
        preview.configure_traits()

    # --- major VGroups
    vgr0 = VGroup(Label("Defined Materials"),
                 UItem('selected_model', label="Foo",
                       editor=InstanceEditor(view='material_view'),
                       visible_when=("selected_model is not None and "
                                     "len(selected_model.materials) > 0")),
                 Item("none_constant", style='readonly', show_label=False,
                      visible_when=("selected_model is not None and "
                                    "len(selected_model.materials) < 1")))
    vgr1 = VGroup(Label("Installed Models"),
                  UItem('models', editor=TabularEditor(
                      show_titles=True, editable=False,
                      selected='selected_model', multi_select=False,
                      adapter=TabularAdapter(columns=[('Models', 'name')]))),
                  vgr0, visible_when='not rerun',
                  show_border=True)
    vgr2 = VGroup(Label("Material Parameters"),
                  UItem('selected_model',
                        editor=InstanceEditor(view='param_view')),
                  show_border=True)
    vgr3 = VGroup(Label("Boundary Parameters"),
                  UItem('selected_model',
                        editor=InstanceEditor(view='solid_path_view'),
                        visible_when=('selected_model is not None and "eos" '
                                      'not in selected_model.model_type')),
                  UItem('selected_model',
                        editor=InstanceEditor(view='eos_boundary_view'),
                        visible_when=('selected_model is not None and "eos" in '
                                      'selected_model.model_type')),
                  show_border=True)
    vgrp = VGroup(HGroup(vgr1, vgr2), vgr3,
                  Item('runid', style="simple"),
                  HGroup(Spring(),
                         Item('show_button', show_label=False,
                              enabled_when="selected_model is not None"),
                         Item('run_button', show_label=False,
                              enabled_when="selected_model is not None"),
                         show_border=True))

    traits_view = View(vgrp, style='custom',
                       width=1024, height=768, resizable=True)


if __name__ == "__main__":
    pm = MMLMaterialModelSelector(
        supplied_data=[('Foo', 'elastic_al_6061.out')])
    pm.configure_traits()
