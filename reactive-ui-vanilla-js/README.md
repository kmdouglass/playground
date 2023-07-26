# Reactive UIs in Vanilla JS

Experiments in reactive (a.k.a. state-based) user interfaces using Vanilla JS.

## Philosophy

[State based UI vs. manual DOM manipulation](https://gomakethings.com/state-based-ui-vs.-manual-dom-manipulation/)

- State is just data with a time-bound aspect to it.
- The approach is to store all the data in a data store (e.g. a JS object) and update the UI whenever certain elements of the data store change.

The approach requires three things:

- A data object.
- A template for how the UI should look based on different data states.
- A function to render the template into the DOM.

### Why?

- Manual DOM manipulations become exceedingly difficult to manage as the size of project grows.

## Additional Links

- [State-based UI with vanilla JS](https://gomakethings.com/state-based-ui-with-vanilla-js/)
- [Simple reactive data stores with vanilla JavaScript and Proxies](https://gomakethings.com/simple-reactive-data-stores-with-vanilla-javascript-and-proxies/)