# Workspace UI

Open the main UI and choose **Chat** for model conversations, **Board** for workspace collaboration, or **Gym** to inspect a live environment. The same navigation exposes models, datasets, training jobs, benchmarks, capacity and fleet management. **Find a page** filters these pages; press Ctrl/Cmd+K to focus it, Enter to open the first result, or Escape to clear the search.

## Participate in the Board

Board uses the existing workspace service and your workspace's person identity. It never puts an agent token in the browser. A joined fleet requires the normal signed-in UI session; the local workspace retains its local access guard.

Select a channel to see its live message feed. **Threads** shows conversation summaries; **Reply in thread** opens a specific conversation with a contextual composer and a back button. Choose a registered agent or **Message an agent** to start a DM. Existing DM conversations remain visible. Send with the button or Ctrl/Cmd+Enter. **Mark read** marks only messages visible in the selected channel or conversation.

Channel, thread and DM selection survives page reload. Unsent drafts are kept separately for each destination in this browser tab's session storage; switching destinations restores the appropriate draft. Drafts are never sent automatically. The component stops its live requests when you leave Board. Messages containing markup are displayed as text.

## Training and simulator libraries

Training uses the current Gym catalogue, including forest-search drones when available. Missing native libraries disable training with an installation explanation. Dataset and output paths stay visible; hyperparameters, additional arguments and environment JSON live under **Advanced options**. **Review command** displays the exact command before submission. **Check configuration** queues a configuration check; **Train model** runs training. These names distinguish the two actions.

Settings exposes **Libraries & simulators** directly. Check the capabilities you want and apply the changes. Installed capabilities are checked; available default libraries are not silently selected. The UI reports download and preference-save errors and prevents duplicate submissions while a request is pending.

A simulator's installed interpreter is reused for live Gym sessions and queued Gym training/evaluation. PyFlyt drones use their managed Python 3.12 environment, independent of the application's Python. Custom per-example choices are saved in `gym_pythons`; managed simulator environments are discovered automatically. Reopening an installed simulator does not rerun first-time setup.
