// Command ovdevices prints the OpenVINO build and the devices it can see.
package main

import (
	"fmt"

	"github.com/daaku/openvino"
)

func main() {
	b, d, err := openvino.Version()
	fmt.Println(b, d, err)
	c, err := openvino.SharedCore()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(c.AvailableDevices())
}
