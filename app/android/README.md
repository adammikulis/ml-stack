# Native Android companion

The companion scans or pastes an Android enrollment invite, verifies the computer certificate,
then stores its scoped credential encrypted in Android Keystore. Android's strong biometric or
device credential prompt unlocks the connection. Leaving the app closes requests and clears
its unlocked session. Removing the saved connection requires a new invite.

Create an Android invite in Fleet on the computer's local owner interface. Approving that
invite grants one hour of local model chat and status. Fleet also lists and revokes Android
connections. Restarting the computer daemon, changing cluster membership or its listener
certificate invalidates the connection. Computer enrollment invites are rejected. The
companion never receives a cluster execution key. Replies stream as generated; Cancel and
leaving the app close the request.

The app supports Android R and later. Camera permission is requested only for scanning; paste
works without it. No images or biometric information are stored. Release metadata comes from
the repository's owner-managed Python release metadata.

Build from this directory with an installed Android SDK and JDK. Select a stable Android Gradle
plugin through `-PandroidPlugin`, and the installed platform through `-PandroidApi`. Supply
`ANDROID_SIGNING_STORE` and `ANDROID_SIGNING_PASSWORD` for a signing keystore whose alias is
`ml-stack-preview`. The maintained CPU broker must admit builds and protocol tests.

Android SDK downloads require a person to accept the current SDK license at
https://developer.android.com/studio#downloads before downloading the command-line tools.
The selected SDK packages prompt for person approval during setup; do not run an automatic blanket license
acceptance. Keep SDK, Gradle caches, and preview signing material outside the checkout.

Run `build.ps1` with an existing SDK to create the signed release APK. The protocol checks run
before Android compilation. A build verifies compilation and packaging; physical-device QR,
biometric, background-lock, and certificate-pinning checks require an Android device.
