class Component(object):
  def __init__(self, inputs, output,
               output_infer=None,
               child_components=None,
               feature_extractor=None, name=None):
    self._inputs = inputs
    self._output = output
    self._output_infer = output_infer if output_infer is not None else output
    self._child_components = child_components
    self._feature_extractor = feature_extractor
    self._name = name

  def name(self):
    return self._name

  @property
  def output(self):
    return self._output

  @property
  def output_infer(self):
    return self._output_infer

  def _input_feeds(self, **raw_input):
    if not self._inputs:
      return {}
    values = self._feature_extractor.extract_features(**raw_input)
    assert (len(values) == len(self._inputs))
    return {var: val for var, val in zip(self._inputs, values)}

  def input_feeds(self, **raw_input):
    feeds = self._input_feeds(**raw_input)
    if self._child_components:
      for c in self._child_components:
        feeds.update(c.input_feeds(**raw_input))
    return feeds
